import xdrl.semantics as semantics


def test_semantics_exports_only_public_contracts() -> None:
    assert "AgentIdentity" in semantics.__all__
    assert "InternalCoordinate" in semantics.__all__
    assert "_key_path" not in semantics.__all__
