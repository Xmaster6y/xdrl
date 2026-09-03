from __future__ import annotations

import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from setuptools.build_meta import build_wheel

ROOT = Path(__file__).parents[2]
SKILL = ROOT / "skills" / "xdrl-library" / "SKILL.md"


def _runnable_examples(markdown: str) -> list[str]:
    return re.findall(r"<!-- runnable-example -->\s*```python\n(.*?)```", markdown, re.DOTALL)


@pytest.mark.integration
def test_skill_examples_use_the_installed_public_api(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / build_wheel(str(dist))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)

    examples = _runnable_examples(SKILL.read_text())
    assert len(examples) == 1
    for index, example in enumerate(examples):
        script = tmp_path / f"example_{index}.py"
        script.write_text(
            "from pathlib import Path\n"
            "import xdrl\n"
            f"assert Path(xdrl.__file__).is_relative_to(Path({str(installed)!r}))\n" + example
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(installed)
        environment["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )


def test_library_skill_describes_the_current_public_boundary() -> None:
    skill = SKILL.read_text()

    assert "component.run(workflow, data)" in skill
    assert "Target(occurrences=(...,))" in skill
    assert "TDHook's native `WorkflowResult`" in skill
