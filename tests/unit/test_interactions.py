import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from xdrl import (
    BatchSemantics,
    Interaction,
    InteractionSpec,
    KeyRole,
    KeySchema,
    ModelRole,
    SchemaValidationError,
    TensorDictSchema,
)


def _interaction(module: torch.nn.Module | None = None, **modes: object) -> Interaction:
    policy = TensorDictModule(
        torch.nn.Linear(2, 1, bias=False) if module is None else module,
        in_keys=["observation"],
        out_keys=["action"],
    )
    spec = InteractionSpec(
        ModelRole.ACTOR,
        TensorDictSchema((KeySchema("observation", KeyRole.OBSERVATION),)),
        TensorDictSchema((KeySchema("action", KeyRole.ACTION),)),
        BatchSemantics(("env",)),
        **modes,
    )
    return Interaction(policy, spec)


def test_interaction_runs_an_unchanged_torchrl_module() -> None:
    interaction = _interaction()
    data = TensorDict({"observation": torch.ones(3, 2)}, batch_size=[3])

    result = interaction(data)

    assert result["action"].shape == (3, 1)
    assert not hasattr(interaction.module, "input_schema")


def test_interaction_validates_both_boundaries() -> None:
    interaction = _interaction()
    with pytest.raises(SchemaValidationError, match="interaction input"):
        interaction(TensorDict({}, batch_size=[2]))

    valid = _interaction(torch.nn.Identity())
    invalid = Interaction(
        valid.module,
        InteractionSpec(
            ModelRole.ACTOR,
            valid.spec.inputs,
            TensorDictSchema((KeySchema("missing_action", KeyRole.ACTION),)),
            valid.spec.batch,
        ),
    )
    with pytest.raises(SchemaValidationError, match="interaction output"):
        invalid(TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2]))


class _ModeProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.child = torch.nn.Identity()
        self.seen: list[tuple[bool, bool]] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.seen.append((self.training, torch.is_grad_enabled()))
        return value[:, :1]


def test_interaction_scopes_and_restores_training_and_gradient_modes() -> None:
    probe = _ModeProbe()
    interaction = _interaction(probe, training=False, gradient_enabled=False)
    interaction.module.train()
    probe.child.eval()

    interaction(TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2]))

    assert probe.seen == [(False, False)]
    assert interaction.module.training and probe.training
    assert not probe.child.training


def test_interaction_restores_modes_after_failure() -> None:
    class Failing(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("boom")

    interaction = _interaction(Failing(), training=False)
    interaction.module.train()

    with pytest.raises(RuntimeError, match="boom"):
        interaction(TensorDict({"observation": torch.ones(2, 2)}, batch_size=[2]))
    assert interaction.module.training


def test_interaction_spec_rejects_conflicting_modes() -> None:
    with pytest.raises(ValueError, match="cannot both"):
        _interaction(gradient_enabled=True, inference_mode=True)
