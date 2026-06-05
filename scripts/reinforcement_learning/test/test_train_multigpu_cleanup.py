# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the multi-GPU training launcher cleanup behavior."""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_MULTIGPU_SCRIPT = REPO_ROOT / "scripts" / "reinforcement_learning" / "train_multigpu.py"


def _load_train_multigpu_module():
    spec = importlib.util.spec_from_file_location("train_multigpu", TRAIN_MULTIGPU_SCRIPT)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_run_distributed_command_cleans_up_process_group_children(tmp_path: Path) -> None:
    """The launcher must not leave distributed worker descendants running after the parent exits."""
    train_multigpu = _load_train_multigpu_module()
    child_pid_file = tmp_path / "child.pid"
    helper_script = tmp_path / "spawn_child.py"
    helper_script.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import subprocess",
                "import sys",
                "",
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                "with open(sys.argv[1], 'w', encoding='utf-8') as stream:",
                "    stream.write(str(child.pid))",
            ]
        ),
        encoding="utf-8",
    )

    returncode = train_multigpu._run_distributed_command(
        [sys.executable, str(helper_script), str(child_pid_file)]
    )

    assert returncode == 0
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not _pid_exists(child_pid):
            break
        time.sleep(0.1)

    if _pid_exists(child_pid):
        os.kill(child_pid, signal.SIGKILL)
        subprocess.run(["ps", "-o", "pid,ppid,pgid,stat,command", "-p", str(child_pid)], check=False)
        raise AssertionError(f"launcher left child process running: pid={child_pid}")
