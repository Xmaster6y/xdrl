"""Refresh and validate XDRL's test-backed dependency snapshot."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import NamedTuple

from packaging.markers import Marker, UndefinedEnvironmentName


ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = ROOT / "uv.lock"
PYPROJECT = ROOT / "pyproject.toml"
COMPATIBILITY = ROOT / "src" / "xdrl" / "compatibility.py"
COMPATIBILITY_START = "# dependency-snapshot: start"
COMPATIBILITY_END = "# dependency-snapshot: end"
DOCUMENTATION = ROOT / "docs" / "source" / "compatibility.rst"
DOCUMENTATION_START = ".. dependency-snapshot: start"
DOCUMENTATION_END = ".. dependency-snapshot: end"

TRACKED_DEPENDENCIES = ("torch", "tensordict", "torchrl", "tdhook", "xdrl")
REFRESH_PACKAGES = ("torch", "tensordict", "torchrl", "tdhook")
EVIDENCE = {
    "torch": "compatibility and parity suites",
    "tensordict": "schema and parity suites",
    "torchrl": "compatibility and integration",
    "tdhook": "adapter conformance suite",
    "xdrl": "all required suites",
}


class SnapshotError(RuntimeError):
    """The lockfile cannot produce one unambiguous compatibility snapshot."""


class Dependency(NamedTuple):
    """One exact distribution selected by the universal lockfile."""

    name: str
    version: str
    commit: str | None
    marker: str | None = None


def read_snapshot(lockfile: Path = LOCKFILE) -> tuple[Dependency, ...]:
    """Read exact versions and Git revisions from ``uv.lock``."""
    payload = tomllib.loads(lockfile.read_text())
    packages = payload.get("package", [])
    snapshot: list[Dependency] = []
    for name in TRACKED_DEPENDENCIES:
        matches = [package for package in packages if package.get("name") == name]
        if not matches:
            raise SnapshotError(f"expected at least one {name!r} package in {lockfile}")
        for package in matches:
            version = package.get("version")
            if not isinstance(version, str) or not version:
                raise SnapshotError(f"package {name!r} has no exact version in {lockfile}")
            source = package.get("source", {})
            git = source.get("git") if isinstance(source, dict) else None
            commit = _git_commit(git, name) if isinstance(git, str) else None
            marker = _resolution_marker(package.get("resolution-markers"), name)
            snapshot.append(Dependency(name, version, commit, marker))
    return tuple(snapshot)


def compatibility_block(snapshot: tuple[Dependency, ...]) -> str:
    """Render the generated public compatibility declarations."""
    requirements: list[str] = []
    for dependency in snapshot:
        if dependency.marker is None:
            requirements.append(f'    VersionRequirement("{dependency.name}", "{_specifier(dependency)}"),')
        else:
            requirements.extend(
                (
                    "    VersionRequirement(",
                    f'        "{dependency.name}",',
                    f'        "{_specifier(dependency)}",',
                    f"        marker={dependency.marker!r},",
                    "    ),",
                )
            )
    revisions = [
        f'    GitRevisionRequirement("{dependency.name}", "{dependency.commit}"),'
        for dependency in snapshot
        if dependency.commit is not None
    ]
    return "\n".join(
        [
            COMPATIBILITY_START,
            "SUPPORTED_DEPENDENCIES = (",
            *requirements,
            ")",
            "SUPPORTED_GIT_REVISIONS = (",
            *revisions,
            ")",
            COMPATIBILITY_END,
        ]
    )


def documentation_block(snapshot: tuple[Dependency, ...]) -> str:
    """Render the generated compatibility matrix rows."""
    rows = [("Python", "``>=3.11,<3.14``", "required CI matrix")]
    display_names = {
        "torch": "PyTorch",
        "tensordict": "TensorDict",
        "torchrl": "TorchRL",
        "tdhook": "TDHook",
        "xdrl": "xdrl",
    }
    rows.extend(
        (
            display_names[item.name],
            f"``{_specifier(item)}``",
            EVIDENCE[item.name] + (f"; ``{item.marker}``" if item.marker is not None else ""),
        )
        for item in snapshot
    )
    widths = [max(len(row[column]) for row in rows) for column in range(3)]
    separator = "  ".join("=" * width for width in widths)
    body = [DOCUMENTATION_START, "", separator]
    body.append(
        "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(("Component", "Tested requirement", "Evidence"))
        ).rstrip()
    )
    body.append(separator)
    for component, requirement, evidence in rows:
        values = (component, requirement, evidence)
        body.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(values)).rstrip())
    body.extend((separator, "", DOCUMENTATION_END))
    return "\n".join(body)


def synchronize(snapshot: tuple[Dependency, ...], *, check: bool) -> list[Path]:
    """Update or check all generated views of the lockfile snapshot."""
    replacements = {
        COMPATIBILITY: (COMPATIBILITY_START, COMPATIBILITY_END, compatibility_block(snapshot)),
        DOCUMENTATION: (DOCUMENTATION_START, DOCUMENTATION_END, documentation_block(snapshot)),
    }
    changed: list[Path] = []
    for path, (start, end, block) in replacements.items():
        original = path.read_text()
        updated = _replace_block(original, start, end, block, path)
        if updated != original:
            changed.append(path)
            if not check:
                path.write_text(updated)
    return changed


def report(old: tuple[Dependency, ...], new: tuple[Dependency, ...]) -> str:
    """Return a review-oriented old/new snapshot table."""
    old_by_name = _group_snapshot(old)
    new_by_name = _group_snapshot(new)
    lines = [
        "| Dependency | Old version | New version | Old revision | New revision |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in TRACKED_DEPENDENCIES:
        previous = old_by_name[name]
        current = new_by_name[name]
        lines.append(
            f"| {name} | {_display_versions(previous)} | {_display_versions(current)} | "
            f"{_display_revisions(previous)} | {_display_revisions(current)} |"
        )
    return "\n".join(lines)


def refresh() -> int:
    """Resolve configured sources, synchronize declarations, and report the delta."""
    validate_refresh_sources()
    validate_pytorch_sources()
    old = read_snapshot()
    command = ["uv", "lock"]
    for package in REFRESH_PACKAGES:
        command.extend(("--upgrade-package", package))
    subprocess.run(command, cwd=ROOT, check=True)
    new = read_snapshot()
    synchronize(new, check=False)
    summary = report(old, new)
    print(summary)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a") as stream:
            stream.write("## Dependency snapshot\n\n")
            stream.write(summary)
            stream.write("\n")
    return 0


def check() -> int:
    """Fail when the lock, declarations, or documented matrix disagree."""
    validate_refresh_sources()
    validate_pytorch_sources()
    subprocess.run(("uv", "lock", "--check"), cwd=ROOT, check=True)
    changed = synchronize(read_snapshot(), check=True)
    if changed:
        paths = ", ".join(str(path.relative_to(ROOT)) for path in changed)
        print(f"dependency snapshot drift detected in: {paths}", file=sys.stderr)
        print("run `just sync-dependency-snapshot` after reviewing uv.lock", file=sys.stderr)
        return 1
    print("dependency snapshot is consistent")
    return 0


def validate_refresh_sources(pyproject: Path = PYPROJECT) -> None:
    """Reject Git source constraints that cannot advance during a refresh."""
    payload = tomllib.loads(pyproject.read_text())
    sources = payload.get("tool", {}).get("uv", {}).get("sources", {})
    for package in REFRESH_PACKAGES:
        source = sources.get(package)
        if not isinstance(source, dict) or "git" not in source:
            continue
        revision = source.get("rev")
        if isinstance(revision, str) and _is_commit(revision):
            raise SnapshotError(
                f"refresh-managed Git source {package!r} is pinned to an immutable commit; track a branch instead"
            )


def validate_pytorch_sources(pyproject: Path = PYPROJECT) -> None:
    """Prove optional stable/nightly markers cover the supported matrix once."""
    payload = tomllib.loads(pyproject.read_text())
    uv = payload.get("tool", {}).get("uv", {})
    sources = uv.get("sources", {})
    torch_sources = sources.get("torch")
    if torch_sources is None or isinstance(torch_sources, dict):
        return
    if not isinstance(torch_sources, list) or not torch_sources:
        raise SnapshotError("tool.uv.sources.torch must be a source or a non-empty marker list")
    indexes = {index.get("name") for index in uv.get("index", []) if isinstance(index, dict)}
    for source in torch_sources:
        if not isinstance(source, dict) or not isinstance(source.get("marker"), str):
            raise SnapshotError("every platform-specific PyTorch source must declare a marker")
        if source.get("index") not in indexes:
            raise SnapshotError(f"PyTorch source refers to undeclared index {source.get('index')!r}")
    for python_version in ("3.11", "3.12", "3.13"):
        for sys_platform in ("darwin", "linux", "win32"):
            environment = {"python_version": python_version, "sys_platform": sys_platform}
            try:
                matches = [source for source in torch_sources if Marker(source["marker"]).evaluate(environment)]
            except UndefinedEnvironmentName as error:
                raise SnapshotError(f"PyTorch source marker cannot be evaluated: {error}") from error
            if len(matches) != 1:
                raise SnapshotError(
                    "PyTorch source markers must select exactly one index for "
                    f"Python {python_version} on {sys_platform}; selected {len(matches)}"
                )


def _git_commit(source: str, name: str) -> str:
    _, separator, commit = source.rpartition("#")
    if not separator or not _is_commit(commit):
        raise SnapshotError(f"Git package {name!r} has no immutable 40-character commit in uv.lock")
    return commit


def _is_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value.lower())


def _resolution_marker(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(marker, str) for marker in value):
        raise SnapshotError(f"package {name!r} has invalid resolution markers in uv.lock")
    return " or ".join(f"({marker})" for marker in value)


def _specifier(dependency: Dependency) -> str:
    if dependency.name == "torch":
        release = dependency.version.split("+", 1)[0].split(".")
        if len(release) < 2:
            raise SnapshotError(f"cannot derive a minor PyTorch boundary from {dependency.version!r}")
        return f"=={release[0]}.{release[1]}.*"
    return f"=={dependency.version}"


def _replace_block(text: str, start: str, end: str, replacement: str, path: Path) -> str:
    start_index = text.find(start)
    end_index = text.find(end, start_index + len(start))
    if (
        start_index < 0
        or end_index < 0
        or text.find(start, start_index + 1) >= 0
        or text.find(end, end_index + 1) >= 0
    ):
        raise SnapshotError(f"expected exactly one generated snapshot block in {path}")
    end_index += len(end)
    return text[:start_index] + replacement + text[end_index:]


def _short_revision(commit: str | None) -> str:
    return commit[:12] if commit is not None else "registry"


def _group_snapshot(snapshot: tuple[Dependency, ...]) -> dict[str, tuple[Dependency, ...]]:
    return {name: tuple(item for item in snapshot if item.name == name) for name in TRACKED_DEPENDENCIES}


def _display_versions(dependencies: tuple[Dependency, ...]) -> str:
    return "<br>".join(
        dependency.version + (f" ({dependency.marker})" if dependency.marker is not None else "")
        for dependency in dependencies
    )


def _display_revisions(dependencies: tuple[Dependency, ...]) -> str:
    return "<br>".join(_short_revision(dependency.commit) for dependency in dependencies)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "refresh", "sync"))
    arguments = parser.parse_args(argv)
    if arguments.action == "check":
        return check()
    if arguments.action == "refresh":
        return refresh()
    changed = synchronize(read_snapshot(), check=False)
    for path in changed:
        print(f"updated {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
