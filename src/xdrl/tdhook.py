"""TDHook instrumentation for one typed TorchRL interaction.

The adapter leaves TensorDict execution owned by TorchRL.  It only prepares
TDHook contexts around a selected, already-materialised TensorDict module and
routes the call through :class:`RuntimeInteractionContext` for contract and
lifecycle handling.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tdhook.contexts import HookingContext, HookingContextFactory
from tdhook.hooks import resolve_submodule_path

from xdrl.interactions import RuntimeInteractionContext
from xdrl.types import KeyPresence, TensorDictKey


def _key_path(key: TensorDictKey) -> tuple[str, ...]:
    return tuple(str(part) for part in key) if isinstance(key, tuple) else (str(key),)


def _module_keys(module: TensorDictModuleBase, attribute: str) -> set[tuple[str, ...]]:
    return {_key_path(key) for key in getattr(module, attribute)}


@dataclass(slots=True)
class TDHookInteractionAdapter:
    """Bind a runtime interaction to raw TDHook context factories.

    ``aliases`` maps a stable semantic name to a path relative to
    ``selected_module`` (for example ``{"encoder": "module.0"}``).  The
    :attr:`target_paths` property exposes the corresponding TDHook target path.
    Factories retain their normal TDHook API; entered contexts are available in
    :attr:`contexts` while the adapter is active.
    """

    interaction: RuntimeInteractionContext
    selected_module: TensorDictModuleBase | None = None
    input_keys: Sequence[TensorDictKey] | None = None
    output_keys: Sequence[TensorDictKey] | None = None
    aliases: Mapping[str, str] = field(default_factory=dict)
    contexts: tuple[HookingContext, ...] = field(default=(), init=False)
    _stack: ExitStack | None = field(default=None, init=False, repr=False)
    _hooked_module: TensorDictModuleBase | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.selected_module is None:
            self.selected_module = self.interaction.module
        if not self.aliases:
            self.aliases = dict(self.interaction.descriptor.module_aliases)
        _reject_unsupported_module(self.selected_module)
        self.input_keys = tuple(self.selected_module.in_keys if self.input_keys is None else self.input_keys)
        self.output_keys = tuple(self.selected_module.out_keys if self.output_keys is None else self.output_keys)
        self._validate_selection()
        self._validate_aliases()

    @property
    def target_paths(self) -> dict[str, str]:
        """Return TDHook paths for semantic aliases, relative to this adapter."""
        return {alias: _tdhook_path(path) for alias, path in self.aliases.items()}

    @property
    def module_paths(self) -> tuple[str, ...]:
        """Discover hookable submodule paths relative to ``selected_module``."""
        assert self.selected_module is not None
        return tuple(name for name, _ in self.selected_module.named_modules() if name)

    def materialize(self, tensordict: TensorDictBase | None = None) -> None:
        """Materialise lazy parameters explicitly using a representative input.

        A lazy module is never silently initialised during :meth:`activate`.
        The normal module call is intentional: callers can choose the exact
        representative batch and execution mode used for materialisation.
        """
        assert self.selected_module is not None
        if not _has_uninitialized_parameters(self.selected_module):
            return
        batch = self.interaction.representative_input if tensordict is None else tensordict
        self.interaction.input_schema.validate_inputs(batch)
        self.selected_module(batch.clone())
        if _has_uninitialized_parameters(self.selected_module):
            raise RuntimeError("selected module remains lazy after materialisation")

    def activate(self, *factories: HookingContextFactory) -> TDHookInteractionAdapter:
        """Enter TDHook contexts and return this adapter as a context manager.

        Contexts are deliberately separate instead of being wrapped in a new
        abstraction, so callers can use factory-specific capabilities such as
        activation caches directly through :attr:`contexts`.
        """
        if self._stack is not None:
            raise RuntimeError("TDHook interaction adapter is already active")
        if not factories:
            raise ValueError("activate requires at least one TDHook context factory")
        assert self.selected_module is not None
        if _has_uninitialized_parameters(self.selected_module):
            raise RuntimeError("selected module has lazy parameters; call materialize() before activate()")

        stack = ExitStack()
        try:
            stack.enter_context(self.interaction)
            contexts = tuple(
                factory.prepare(self.selected_module, list(self.input_keys), list(self.output_keys))
                for factory in factories
            )
            hooked_modules = tuple(stack.enter_context(context) for context in contexts)
        except BaseException:
            stack.close()
            raise
        self.contexts = contexts
        # Every raw context hooks the same selected module.  Any returned
        # HookedModule has the same TensorDict call contract, so one is enough
        # to execute while all context handles remain installed.
        self._hooked_module = hooked_modules[-1]
        self._stack = stack
        return self

    def __enter__(self) -> TDHookInteractionAdapter:
        if self._stack is None:
            raise RuntimeError("use 'with adapter.activate(factory)' to enter TDHook instrumentation")
        return self

    def __exit__(self, *exc_info: object) -> bool | None:
        if self._stack is None:
            return None
        stack, self._stack = self._stack, None
        self._hooked_module, self.contexts = None, ()
        return stack.__exit__(*exc_info)

    def invoke(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Invoke the selected model through TDHook and the interaction contract."""
        if self._hooked_module is None:
            raise RuntimeError("invoke must be called inside an active TDHook adapter")
        return self.interaction.invoke(tensordict, module=self._hooked_module)

    def _validate_selection(self) -> None:
        assert self.selected_module is not None
        model_inputs = _module_keys(self.selected_module, "in_keys")
        model_outputs = _module_keys(self.selected_module, "out_keys")
        selected_inputs = {_key_path(key) for key in self.input_keys}
        selected_outputs = {_key_path(key) for key in self.output_keys}
        _require_subset("selected input", selected_inputs, model_inputs)
        _require_subset("selected output", selected_outputs, model_outputs)
        contract_inputs = {
            _key_path(entry.key)
            for entry in self.interaction.input_schema.keys
            if entry.presence in {KeyPresence.REQUIRED, KeyPresence.OPTIONAL}
        }
        contract_outputs = {
            _key_path(entry.key)
            for entry in self.interaction.output_schema.keys
            if entry.presence is KeyPresence.PRODUCED
        }
        _require_subset("interaction input contract", contract_inputs, selected_inputs)
        _require_subset("interaction output contract", contract_outputs, selected_outputs)

    def _validate_aliases(self) -> None:
        assert self.selected_module is not None
        for alias, path in self.aliases.items():
            if not alias or not isinstance(alias, str):
                raise ValueError("TDHook alias names must be non-empty strings")
            if not isinstance(path, str):
                raise ValueError(f"TDHook alias {alias!r} must map to a string path")
            try:
                resolve_submodule_path(self.selected_module, path)
            except ValueError as error:
                raise ValueError(f"TDHook alias {alias!r} cannot resolve {path!r}") from error


def _require_subset(label: str, actual: set[tuple[str, ...]], expected: set[tuple[str, ...]]) -> None:
    missing = actual - expected
    if missing:
        display = ", ".join("/".join(key) for key in sorted(missing))
        raise ValueError(f"{label} keys are not exposed by the selected module: {display}")


def _tdhook_path(path: str) -> str:
    return "td_module" if not path else f"td_module.{path}"


def _has_uninitialized_parameters(module: torch.nn.Module) -> bool:
    return any(isinstance(parameter, torch.nn.parameter.UninitializedParameter) for parameter in module.parameters())


def _reject_unsupported_module(module: TensorDictModuleBase) -> None:
    if hasattr(module, "_orig_mod"):
        raise NotImplementedError("torch.compile modules are not supported by the TDHook adapter")
    distributed_types = (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)
    if any(isinstance(child, distributed_types) for child in module.modules()):
        raise NotImplementedError("distributed modules are not supported by the TDHook adapter")
    if type(module).__module__.startswith("torch.distributed.rpc"):
        raise NotImplementedError("remote modules are not supported by the TDHook adapter")
