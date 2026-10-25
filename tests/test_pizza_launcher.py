"""The pizza agent launcher selects startup mode, loads the pizza guide, and keeps the task."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "arguments",
    [[], ["--allow-motion"], ["--allow-motion", "exec", "Make the pizza."]],
)
def test_launcher_selects_startup_mode_and_preserves_task(tmp_path, arguments):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / "start_pizza_agent.sh"
    shutil.copyfile(root / "scripts/start_pizza_agent.sh", launcher)
    docs = tmp_path / "docs"
    docs.mkdir()
    guide = (root / "docs/pizza-agent-guide.md").read_text().rstrip("\n")
    (docs / "pizza-agent-guide.md").write_text(guide)
    binaries = tmp_path / ".venv-system2/bin"
    binaries.mkdir(parents=True)
    python = binaries / "python"  # stubs the HTTP health check
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    codex = binaries / "codex"  # captures argv instead of starting an agent
    codex.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    codex.chmod(0o755)

    launched = subprocess.run(
        ["bash", str(launcher), *arguments],
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=True,
    )
    argv = json.loads(launched.stdout)
    prompt = argv[-1]
    motion = "--allow-motion" in arguments
    assert argv[argv.index("-C") + 1] == str(tmp_path)
    assert guide in prompt
    assert ("This is a motion-enabled startup." in prompt) == motion
    assert ("This is an observation-only startup." in prompt) != motion
    approvals = [a for a in argv if "approval_mode" in a]
    assert any("cap_move_to_pose" in a for a in approvals) == motion
    assert any("cap_localize_object" in a for a in approvals)  # read-only tools are always approved
    assert not any("cap_move_synchronized" in a for a in approvals)  # one arm: not offered
    assert prompt.endswith(
        "Make the pizza."
        if "exec" in arguments
        else "After startup completes, wait for the operator task."
    )
    assert ("exec" in argv) == ("exec" in arguments)
