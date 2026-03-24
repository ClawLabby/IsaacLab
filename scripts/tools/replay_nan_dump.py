#!/usr/bin/env python3
"""Replay a NaN watchdog dump through Newton to reproduce physics failures.

Loads the dumped Newton state (previous step), applies the dumped control
actions, steps the solver, and checks if the result contains NaN.

Usage:
    python replay_nan_dump.py /path/to/nan_dumps/nan_stepN_TIMESTAMP/

This will:
1. Load the Newton model from the original task config
2. Restore the "previous" state from the dump (state before NaN)
3. Apply the dumped control actions
4. Step the Newton solver
5. Check if the output state contains NaN
6. If NaN found, report which bodies/joints are affected

For visualization, use Newton's ViewerGL:
    python replay_nan_dump.py /path/to/dump/ --visualize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def load_dump(dump_path: Path) -> dict:
    """Load all files from a NaN dump directory."""
    dump = {}

    # NaN metadata
    info_path = dump_path / "nan_info.json"
    if info_path.exists():
        with open(info_path) as f:
            dump["nan_info"] = json.load(f)

    # Newton state
    state_path = dump_path / "newton_state.pt"
    if state_path.exists():
        dump["newton_state"] = torch.load(state_path, weights_only=False)
    else:
        print("WARNING: No newton_state.pt found. Was Newton running?")

    # RL tensors
    rl_path = dump_path / "rl_tensors.pt"
    if rl_path.exists():
        dump["rl_tensors"] = torch.load(rl_path, weights_only=False)

    return dump


def inspect_dump(dump: dict):
    """Print summary of what's in the dump."""
    print("=" * 60)
    print("NaN Dump Inspection")
    print("=" * 60)

    if "nan_info" in dump:
        info = dump["nan_info"]
        print(f"\nStep: {info['step']}")
        print(f"Timestamp: {info['timestamp']}")
        print(f"Dump number: {info['dump_number']}")
        for loc, details in info.get("nan_locations", {}).items():
            print(f"\n  {loc}:")
            print(f"    NaN count: {details.get('count', '?')} / {details.get('total', '?')}")
            print(f"    Shape: {details.get('shape', '?')}")
            envs = details.get("env_indices", [])
            print(f"    Affected envs ({len(envs)}): {envs[:20]}{'...' if len(envs) > 20 else ''}")

    if "newton_state" in dump:
        ns = dump["newton_state"]
        print(f"\nNewton State Keys: {list(ns.keys())}")

        # Check for NaN in state tensors
        for key, val in ns.items():
            if isinstance(val, torch.Tensor):
                nan_count = val.isnan().sum().item()
                inf_count = val.isinf().sum().item()
                if nan_count > 0 or inf_count > 0:
                    print(f"  ⚠️  {key}: {nan_count} NaN, {inf_count} Inf (shape={val.shape})")
                else:
                    print(f"  ✓ {key}: clean (shape={val.shape}, range=[{val.min():.4f}, {val.max():.4f}])")
            elif isinstance(val, dict):
                print(f"  {key}: {val}")
            else:
                print(f"  {key}: {val}")

        # Check previous state (this is what we'd use for replay)
        has_previous = any(k.startswith("previous.") for k in ns.keys())
        has_control = any(k.startswith("control.") for k in ns.keys())
        print(f"\n  Has previous state: {has_previous}")
        print(f"  Has control actions: {has_control}")
        if has_previous and has_control:
            print("  → Replay possible: load previous state, apply control, step solver")
        elif has_previous:
            print("  → Partial replay: have state but no control actions")
        else:
            print("  → Cannot replay: no previous state")

    if "rl_tensors" in dump:
        rl = dump["rl_tensors"]
        print(f"\nRL Tensors:")
        for key, val in rl.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    nan_count = v.isnan().sum().item() if isinstance(v, torch.Tensor) else 0
                    print(f"  {key}.{k}: shape={v.shape}, NaN={nan_count}")
            elif isinstance(val, torch.Tensor):
                nan_count = val.isnan().sum().item()
                print(f"  {key}: shape={val.shape}, NaN={nan_count}")
            elif val is None:
                print(f"  {key}: None")


def attempt_replay(dump: dict, device: str = "cpu"):
    """Try to replay the physics step that produced NaN.

    This requires the Newton model to be reconstructed from the task config,
    which is task-specific. This function demonstrates the approach.
    """
    ns = dump.get("newton_state", {})

    has_previous = any(k.startswith("previous.") for k in ns.keys())
    has_control = any(k.startswith("control.") for k in ns.keys())

    if not has_previous:
        print("\nCannot replay: no previous state in dump")
        return

    print("\n" + "=" * 60)
    print("Replay Attempt")
    print("=" * 60)

    print(f"\nTo replay this dump through Newton, you need to:")
    print(f"  1. Reconstruct the Newton Model from the task config")
    print(f"  2. Load 'previous.*' tensors into state_in")
    print(f"  3. Load 'control.*' tensors into the control object")
    print(f"  4. Call solver.step(state_in, state_out, control, contacts, dt)")
    print(f"  5. Check state_out for NaN")
    print(f"\nExample code:")
    print(f"""
    import newton
    import torch

    # Load the dump
    dump = torch.load("{dump.get('_path', 'nan_dumps/nan_stepN/newton_state.pt')}", weights_only=False)

    # Reconstruct model (task-specific — use your env config)
    # For DexSuite Kuka Allegro:
    #   builder = newton.ModelBuilder()
    #   ... (load MJCF/USD) ...
    #   model = builder.finalize(device="{device}")

    # Restore previous state
    state_in = model.state()
    state_in.body_q.assign(dump["previous.body_q"].to("{device}"))
    state_in.body_qd.assign(dump["previous.body_qd"].to("{device}"))
    state_in.joint_q.assign(dump["previous.joint_q"].to("{device}"))
    state_in.joint_qd.assign(dump["previous.joint_qd"].to("{device}"))

    # Restore control
    control = model.control()
    if "control.joint_act" in dump:
        control.joint_act.assign(dump["control.joint_act"].to("{device}"))

    # Step
    state_out = model.state()
    solver.step(state_in, state_out, control, None, dt)

    # Check for NaN
    for attr in ['body_q', 'body_qd', 'joint_q', 'joint_qd']:
        val = getattr(state_out, attr)
        if val is not None:
            t = torch.from_numpy(val.numpy())
            if t.isnan().any():
                print(f"NaN in state_out.{{attr}}")
    """)

    # Check if previous state itself has NaN (shouldn't, that's the state BEFORE failure)
    prev_nan = False
    for key, val in ns.items():
        if key.startswith("previous.") and isinstance(val, torch.Tensor) and val.isnan().any():
            print(f"\n⚠️  Previous state has NaN in {key} — physics was already broken before this step")
            prev_nan = True

    if not prev_nan:
        print("\n✓ Previous state is clean — the NaN was introduced during the step")


def main():
    parser = argparse.ArgumentParser(description="Inspect and replay NaN watchdog dumps")
    parser.add_argument("dump_path", type=str, help="Path to nan dump directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device for replay")
    parser.add_argument("--replay", action="store_true", help="Attempt replay")
    args = parser.parse_args()

    dump_path = Path(args.dump_path)
    if not dump_path.exists():
        print(f"Error: {dump_path} does not exist")
        sys.exit(1)

    dump = load_dump(dump_path)
    dump["_path"] = str(dump_path / "newton_state.pt")

    inspect_dump(dump)

    if args.replay:
        attempt_replay(dump, args.device)


if __name__ == "__main__":
    main()
