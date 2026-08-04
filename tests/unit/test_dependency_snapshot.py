from pathlib import Path
import tomllib

import pytest

from scripts.dependency_snapshot import (
    Dependency,
    SnapshotError,
    compatibility_block,
    documentation_block,
    read_snapshot,
    report,
    validate_refresh_sources,
    validate_pytorch_sources,
)


def _write_lock(path: Path, *, tdhook_commit: str = "a" * 40) -> None:
    packages = [
        ("torch", "2.11.0", None),
        ("tensordict", "0.12.2", None),
        ("torchrl", "0.12.0+g12345678", "b" * 40),
        ("tdhook", "0.1.3", tdhook_commit),
        ("xdrl", "0.1.0", None),
    ]
    blocks = ["version = 1", "revision = 3"]
    for name, version, commit in packages:
        source = (
            f'{{ git = "https://example.invalid/{name}?branch=main#{commit}" }}'
            if commit
            else '{ registry = "https://pypi.org/simple" }'
        )
        blocks.append(f'[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {source}')
    path.write_text("\n\n".join(blocks) + "\n")


def _append_conditional_torch(path: Path) -> None:
    text = path.read_text().replace(
        'name = "torch"\nversion = "2.11.0"\nsource = { registry = "https://pypi.org/simple" }',
        'name = "torch"\nversion = "2.11.0"\nsource = { registry = "https://pypi.org/simple" }\n'
        "resolution-markers = [\"python_version == '3.12'\"]",
    )
    text += (
        '\n[[package]]\nname = "torch"\nversion = "2.12.0.dev1+cpu"\n'
        'source = { registry = "https://download.pytorch.org/whl/nightly/cpu" }\n'
        "resolution-markers = [\"python_version != '3.12'\"]\n"
    )
    path.write_text(text)


def test_read_snapshot_extracts_exact_versions_and_git_commits(tmp_path: Path) -> None:
    lockfile = tmp_path / "uv.lock"
    _write_lock(lockfile)

    snapshot = read_snapshot(lockfile)

    assert snapshot[0] == Dependency("torch", "2.11.0", None)
    assert snapshot[2].commit == "b" * 40
    assert snapshot[3].commit == "a" * 40


def test_read_snapshot_rejects_a_non_immutable_git_source(tmp_path: Path) -> None:
    lockfile = tmp_path / "uv.lock"
    _write_lock(lockfile, tdhook_commit="main")

    with pytest.raises(SnapshotError, match="immutable 40-character commit"):
        read_snapshot(lockfile)


def test_read_snapshot_preserves_conditional_stable_and_nightly_versions(tmp_path: Path) -> None:
    lockfile = tmp_path / "uv.lock"
    _write_lock(lockfile)
    _append_conditional_torch(lockfile)

    torch = [item for item in read_snapshot(lockfile) if item.name == "torch"]

    assert [item.version for item in torch] == ["2.11.0", "2.12.0.dev1+cpu"]
    assert torch[0].marker == "(python_version == '3.12')"
    assert torch[1].marker == "(python_version != '3.12')"


def test_generated_views_share_the_same_snapshot() -> None:
    snapshot = (
        Dependency("torch", "2.11.0", None),
        Dependency("tensordict", "0.12.2", None),
        Dependency("torchrl", "0.12.0+g12345678", "b" * 40),
        Dependency("tdhook", "0.1.3", "a" * 40),
        Dependency("xdrl", "0.1.0", None),
    )

    declarations = compatibility_block(snapshot)
    documentation = documentation_block(snapshot)

    assert 'VersionRequirement("torch", "==2.11.*")' in declarations
    assert 'GitRevisionRequirement("torchrl", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")' in declarations
    assert "``==0.12.0+g12345678``" in documentation


def test_report_makes_registry_and_revision_changes_explicit() -> None:
    old = (Dependency("torch", "2.11.0", None), Dependency("torchrl", "0.12.0", "a" * 40))
    new = (Dependency("torch", "2.12.0", None), Dependency("torchrl", "0.13.0", "b" * 40))

    output = report(old, new)

    assert "| torch | 2.11.0 | 2.12.0 | registry | registry |" in output
    assert "| torchrl | 0.12.0 | 0.13.0 | aaaaaaaaaaaa | bbbbbbbbbbbb |" in output


def test_fixture_lock_is_valid_toml(tmp_path: Path) -> None:
    lockfile = tmp_path / "uv.lock"
    _write_lock(lockfile)
    assert len(tomllib.loads(lockfile.read_text())["package"]) == 5


def test_pytorch_stable_and_nightly_markers_cover_the_supported_matrix(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.uv.sources]
torch = [
    { index = "stable", marker = "sys_platform != 'darwin' or python_version == '3.12'" },
    { index = "nightly", marker = "sys_platform == 'darwin' and python_version != '3.12'" },
]
[[tool.uv.index]]
name = "stable"
url = "https://pypi.org/simple"
[[tool.uv.index]]
name = "nightly"
url = "https://download.pytorch.org/whl/nightly/cpu"
"""
    )

    validate_pytorch_sources(pyproject)


def test_pytorch_markers_reject_an_uncovered_platform(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.uv.sources]
torch = [{ index = "stable", marker = "sys_platform == 'linux'" }]
[[tool.uv.index]]
name = "stable"
url = "https://pypi.org/simple"
"""
    )

    with pytest.raises(SnapshotError, match="select exactly one index"):
        validate_pytorch_sources(pyproject)


def test_pytorch_markers_use_the_synthetic_full_python_and_platform_versions(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.uv.sources]
torch = [{ index = "stable", marker = "python_full_version in '3.11.0 3.12.0 3.13.0' and ((sys_platform == 'darwin' and platform_system == 'Darwin') or (sys_platform == 'linux' and platform_system == 'Linux') or (sys_platform == 'win32' and platform_system == 'Windows'))" }]
[[tool.uv.index]]
name = "stable"
url = "https://pypi.org/simple"
"""
    )

    validate_pytorch_sources(pyproject)


def test_refresh_sources_accept_a_movable_git_branch(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.uv.sources]
tdhook = { git = "https://example.invalid/tdhook", branch = "main" }
torchrl = { git = "https://example.invalid/rl", rev = "parity" }
"""
    )

    validate_refresh_sources(pyproject)


def test_refresh_sources_reject_an_immutable_git_revision(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f"""
[tool.uv.sources]
tdhook = {{ git = "https://example.invalid/tdhook", rev = "{"a" * 40}" }}
"""
    )

    with pytest.raises(SnapshotError, match="pinned to an immutable commit"):
        validate_refresh_sources(pyproject)
