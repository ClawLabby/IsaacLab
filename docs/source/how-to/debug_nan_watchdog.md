# Debugging NaN in Isaac Lab Training

This guide shows how to use the NaN watchdog to detect, dump, and debug physics
failures during training.

## Overview

The NaN watchdog (`isaaclab.utils.nan_watchdog.NaNWatchdog`) provides:

1. **Always-on NaN detection** (~70μs overhead per step)
2. **Automatic state dumps** when NaN is detected
3. **Continue-after-NaN** — affected envs are reset, training continues
4. **Offline replay** — reproduce the exact physics failure

## How It Works

```
Environment.step()
       ↓
  NaNWatchdog.check(obs, rewards, dones, actions)
       ↓
  NaN detected? ─No─→ Continue training
       ↓ Yes
  Dump Newton state + RL tensors
       ↓
  Replace NaN with zeros, mark envs done
       ↓
  Continue training (envs will reset)
```

## Dump Contents

Each dump directory contains:

```
nan_step136032_20260323_092356/
├── nan_info.json       # Metadata: step, timestamp, affected envs
├── newton_state.pt     # Newton physics state (current + previous)
└── rl_tensors.pt       # Observations, rewards, dones, actions
```

### newton_state.pt keys

| Key | Description |
|-----|-------------|
| `current.body_q` | Body poses (7D: pos + quat) after NaN |
| `current.body_qd` | Body velocities (6D: lin + ang) after NaN |
| `current.joint_q` | Joint positions after NaN |
| `current.joint_qd` | Joint velocities after NaN |
| `previous.body_q` | Body poses BEFORE the failing step (clean) |
| `previous.body_qd` | Body velocities BEFORE the failing step |
| `previous.joint_q` | Joint positions BEFORE the failing step |
| `previous.joint_qd` | Joint velocities BEFORE the failing step |
| `control.joint_act` | Joint actuation commands |
| `control.joint_target_pos` | Joint position targets |
| `solver_config` | Solver settings (dt, substeps, iterations) |
| `model.*` | Model topology (body_count, joint_count, etc.) |

## Inspecting a Dump

Use the `replay_nan_dump.py` script:

```bash
cd /path/to/IsaacLab
python scripts/tools/replay_nan_dump.py /path/to/nan_dumps/nan_step136032_*/
```

Example output:
```
============================================================
NaN Dump Inspection
============================================================

Step: 136032
Timestamp: 20260323_092356
Dump number: 1

  obs.proprio:
    NaN count: 330480 / 503808
    Shape: [4096, 123]
    Affected envs (50): [0, 1, 2, 3, ...]

Newton State Keys: [...]
  ⚠️  current.body_q: 840480 NaN, 0 Inf
  ✓ previous.body_q: clean (range=[-95.05, 94.64])  ← Joints way outside limits!
  ✓ previous.joint_q: clean (range=[-95.05, 94.60])
  ...

  Has previous state: True
  Has control actions: True
  → Replay possible
```

## Diagnosing the Root Cause

### Common patterns

1. **Previous state clean, current has NaN**
   - NaN introduced during physics step
   - Check solver convergence, contact geometry

2. **Previous joint_q outside limits**
   - Example: `[-95.05, 94.64]` when limits are `[-3.14, 3.14]`
   - Physics exploded before this step, NaN is secondary
   - Look for extreme forces, contact explosions

3. **control.joint_target_pos has NaN**
   - Policy produced NaN actions
   - Check action clipping, gradient health

### Replay the failure

To reproduce the exact failure:

```python
import torch
import newton
from newton.solvers import SolverMuJoCo

# Load dump
dump = torch.load("nan_dumps/nan_step136032_.../newton_state.pt")

# Reconstruct model (task-specific)
builder = newton.ModelBuilder()
# ... build your robot ...
model = builder.finalize(device="cuda:0")

# Restore previous state
state_in = model.state()
state_in.body_q.assign(dump["previous.body_q"].numpy())
state_in.joint_q.assign(dump["previous.joint_q"].numpy())

# Restore control
control = model.control()
if "control.joint_act" in dump:
    control.joint_act.assign(dump["control.joint_act"].numpy())

# Step solver
solver = SolverMuJoCo(model, **dump["solver_config"])
state_out = model.state()
solver.step(state_in, state_out, control, None, dump["solver_config"]["_solver_dt"])

# Check for NaN
for attr in ['body_q', 'joint_q']:
    val = getattr(state_out, attr)
    t = torch.from_numpy(val.numpy())
    if t.isnan().any():
        print(f"NaN reproduced in {attr}!")
```

## Configuration

The watchdog is enabled automatically when `check_for_nan: true` (default) in
the rsl_rl agent config. Configure via environment variables:

```bash
# Custom dump directory
export NAN_WATCHDOG_DIR=/path/to/dumps

# In agent config
nan_watchdog:
  max_dumps: 3  # Stop dumping after 3 (prevent disk fill)
```

## Preventing NaN

Based on common failure modes:

1. **Increase solver iterations** for complex contact scenarios
2. **Add joint limit enforcement** in the environment reset
3. **Clamp actions** before applying to physics
4. **Monitor for early divergence** (joint positions >> limits)
5. **Use curriculum** to gradually increase difficulty

## Example: The March 2026 Overnight Failure

Overnight training crashed at iteration 4322 (22 hours in). Analysis:

1. **Symptom**: NaN in `proprio` observation group
2. **Dump showed**: `previous.joint_q` range `[-95, +95]` (should be ±3)
3. **Root cause**: Physics exploded silently, joints drifted to extreme values
4. **Solution**: Added watchdog to catch NaN early and continue training

After enabling the watchdog:
- NaN detected at step 136032
- 3 dumps captured with full Newton state
- Training continued and recovered within ~12 iterations
- Reward went from -2 → 21 as normalizer re-learned
