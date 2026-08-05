from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import pytest
from setuptools.build_meta import build_wheel

import xdrl


ROOT = Path(__file__).parents[2]
SKILL = ROOT / "skills" / "xdrl-library"
REFERENCE = SKILL / "references" / "xdrl-0.2.md"


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

    examples = _runnable_examples(REFERENCE.read_text())
    assert len(examples) == 3
    for index, example in enumerate(examples):
        script = tmp_path / f"example_{index}.py"
        script.write_text(
            "from pathlib import Path\n"
            "import xdrl\n"
            f"assert Path(xdrl.__file__).is_relative_to(Path({str(installed)!r}))\n"
            f"assert xdrl.__version__ == {xdrl.__version__!r}\n" + example
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


def test_skill_is_versioned_and_names_evidence_boundaries() -> None:
    skill = (SKILL / "SKILL.md").read_text()
    reference = REFERENCE.read_text()

    assert "xdrl.__version__" in skill
    assert f"xdrl=={xdrl.__version__.rsplit('.', 1)[0]}.*" in reference
    assert "Installation alone is not compatibility evidence" in reference
    assert "semantic labels to positional leading TensorDict batch axes" in reference
    assert (
        "tests/unit/test_observations.py::test_trace_is_serialisable_and_observation_only_preserves_model_output"
        in reference
    )
    assert "tests/behavioural_parity" in reference
    assert "compiled" in reference.casefold() and "distributed" in reference.casefold()
    assert "TDHookWorkflowRunner" in reference
    assert "HookSession" in reference
