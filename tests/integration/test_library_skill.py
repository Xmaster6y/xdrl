from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_library_skill_describes_the_current_public_boundary() -> None:
    skill = (ROOT / "skills" / "xdrl-library" / "SKILL.md").read_text()

    assert "run_workflow(interaction, workflow, data)" in skill
    assert "Target(occurrence=...)" in skill
    assert "TDHook's native `WorkflowResult`" in skill
