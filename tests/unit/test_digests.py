import torch

from xdrl.digests import bytes_digest, module_digest, named_tensor_digest, tensor_digest


def test_tensor_digest_tracks_metadata_and_values() -> None:
    values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)

    assert tensor_digest(values) == tensor_digest(values.clone())
    assert tensor_digest(values) != tensor_digest(values + 1)
    assert tensor_digest(values) != tensor_digest(values.float())
    assert tensor_digest(values) != tensor_digest(values.reshape(4))


def test_tensor_digest_supports_dtypes_without_numpy_conversion() -> None:
    values = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    assert tensor_digest(values) == tensor_digest(values.clone())
    assert tensor_digest(values) != tensor_digest(values + 1)


def test_named_tensor_digest_is_order_independent() -> None:
    tensors = [("left", torch.tensor([1.0])), ("right", torch.tensor([2.0]))]

    assert named_tensor_digest(tensors) == named_tensor_digest(reversed(tensors))


def test_module_digest_tracks_state() -> None:
    module = torch.nn.Linear(2, 1)
    original = module_digest(module)

    with torch.no_grad():
        module.weight.add_(1)

    assert module_digest(module) != original


def test_module_digest_ignores_non_tensor_extra_state() -> None:
    class ModuleWithExtraState(torch.nn.Linear):
        def get_extra_state(self) -> dict[str, str]:
            return {"label": self.label}

        def set_extra_state(self, state: dict[str, str]) -> None:
            self.label = state["label"]

    module = ModuleWithExtraState(2, 1)
    module.label = "before"
    original = module_digest(module)

    module.label = "after"
    assert module_digest(module) == original

    with torch.no_grad():
        module.weight.add_(1)
    assert module_digest(module) != original


def test_bytes_digest_is_sha256() -> None:
    assert bytes_digest(b"xdrl") == "7db3c1130889b590b9ce09dbf85dfccced023632089c8e8c215571f1f8416939"
