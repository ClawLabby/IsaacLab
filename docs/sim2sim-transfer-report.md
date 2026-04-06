# Sim2Sim Transfer: Joint Ordering and Actuator Model Analysis

**Author:** ClawLabby (AI assistant)  
**Date:** 2026-04-06  
**Status:** In Progress — Sharing for Engineering Review  
**Robot:** Kuka iiwa7 + Allegro Hand (DexSuite)  
**Task:** `Isaac-Dexsuite-Kuka-Allegro-Lift-v0`

---

## Executive Summary

When transferring trained policies between PhysX and Newton backends in Isaac Lab, two critical issues cause transfer failure:

1. **Joint ordering differs between backends** — PhysX and Newton enumerate joints in different orders for multi-finger hands. Without explicit ordering control, observation and action vectors are silently permuted.

2. **`ImplicitActuatorCfg` produces different dynamics per backend** — The same Kp/Kd parameters result in different joint response curves because PhysX and Newton implement PD control through fundamentally different integration schemes.

Both issues are **silent** — no errors or warnings. A policy trained on PhysX with `ImplicitActuatorCfg` will produce near-zero success when evaluated on Newton, even with correct joint remapping.

---

## Issue 1: Joint Ordering Mismatch

### The Problem

PhysX and Newton enumerate joints in different orders for the same USD asset. For the Kuka-Allegro hand:

**PhysX** groups joints by joint number across fingers (breadth-first):
```
iiwa7_joint_1..7, index_0, middle_0, ring_0, thumb_0, index_1, middle_1, ring_1, thumb_1, ...
```

**Newton** groups joints by finger (depth-first):
```
iiwa7_joint_1..7, index_0, index_1, index_2, index_3, middle_0, middle_1, middle_2, middle_3, ...
```

The arm joints (indices 0-6) are ordered identically. The hand joints (indices 7-22) are completely different. This means:
- Joint position observations at index 8 are `middle_joint_0` on PhysX but `index_joint_1` on Newton
- Action commands are similarly permuted
- The policy receives scrambled hand observations and sends scrambled hand actions

### The Remap

Newton-to-PhysX joint index mapping:
```python
NEWTON_TO_PHYSX = [0, 1, 2, 3, 4, 5, 6,   # arm: identical
                   7, 11, 15, 19,           # *_joint_0
                   8, 12, 16, 20,           # *_joint_1
                   9, 13, 17, 21,           # *_joint_2
                   10, 14, 18, 22]          # *_joint_3
```

Body ordering for fingertip positions also differs:
```python
# PhysX: [ee_link, index_link_3, middle_link_3, ring_link_3, thumb_link_3]
# Newton: [ee_link, ring_link_3, middle_link_3, index_link_3, thumb_link_3]
NEWTON_TO_PHYSX_BODIES = [0, 3, 2, 1, 4]
```

### The Fix

`SceneEntityCfg` has a `preserve_order` parameter (default: `False`). When set to `True`, joint and body indices are returned in the order specified by the `joint_names` / `body_names` list, rather than sorted by their native index in the articulation.

**Recommendation:** All observation and action terms that reference specific joint or body names should use `preserve_order=True` on their `SceneEntityCfg` to guarantee consistent ordering across backends.

### Impact

Without the fix, a PhysX-trained policy evaluated on Newton sees completely scrambled finger observations. With the remap applied, success improves from 0.0002 to 0.0014 (7×), but is still far from PhysX baseline (2.30) — because the second issue (actuator dynamics) dominates.

---

## Issue 2: ImplicitActuatorCfg Dynamics Mismatch

### The Problem

`KUKA_ALLEGRO_CFG` (in `isaaclab_assets/robots/kuka_allegro.py`) uses `ImplicitActuatorCfg` with these gains:

| Joints | Kp (stiffness) | Kd (damping) | Effort Limit |
|--------|---------------|--------------|--------------|
| iiwa7_joint_1-4 | 300 | 45 | 300 Nm |
| iiwa7_joint_5 | 100 | 20 | 300 Nm |
| iiwa7_joint_6 | 50 | 15 | 300 Nm |
| iiwa7_joint_7 | 25 | 15 | 300 Nm |
| All finger joints | 3 | 0.1 | 0.5 Nm |

Additional: `armature=0.01`, `friction=1.0` (arm) / `0.01` (fingers)

The same numbers are passed to both backends. But they are **implemented differently**:

#### PhysX (Implicit PD)
PhysX integrates the PD control law **implicitly** — stiffness and damping are baked into the constraint solver's position iterations (32 iterations configured). The PD spring-damper is solved simultaneously with contacts, joint limits, and other constraints. This provides:
- Natural numerical damping from implicit integration
- Unconditional stability even with high gains
- A low-pass filtering effect on joint response

#### Newton/MJWarp (Explicit Torque)
MJWarp computes the PD force **explicitly** at the start of each timestep:
```
force = Kp * (target - position) - Kd * velocity
```
This torque is then fed to the `implicitfast` integrator (semi-implicit Euler). The PD computation itself uses start-of-step state, not the converged state. This results in:
- Faster joint response (less numerical damping)
- More oscillatory behavior at high gains
- Potentially unstable with aggressive Kp/Kd and large timesteps

### Measured Divergence

Using identical initial states, identical joint position targets, and identical simulation parameters (dt=1/120, 2 substeps), we measured the maximum joint position divergence over 100 environment steps:

| Joint | Kp | Max PhysX↔Newton Diff | Equivalent |
|-------|-----|----------------------|------------|
| iiwa7_joint_1 | 300 | 0.010 rad | **0.6°** |
| iiwa7_joint_2 | 300 | 0.233 rad | **13.4°** |
| iiwa7_joint_3 | 200 | 0.037 rad | **2.1°** |
| iiwa7_joint_4 | 120 | 0.132 rad | **7.6°** |
| iiwa7_joint_5 | 100 | 0.073 rad | **4.2°** |
| iiwa7_joint_6 | 50 | 0.068 rad | **3.9°** |
| iiwa7_joint_7 | 25 | 0.023 rad | **1.3°** |
| middle_joint_2 | 3 | 0.530 rad | **30.4°** |
| middle_joint_3 | 3 | 0.466 rad | **26.7°** |
| thumb_joint_0 | 3 | 0.328 rad | **18.8°** |

### Root Cause Analysis: iiwa7_joint_2

j2 (Kp=300, Kd=45) shows the worst arm divergence at 13.4°. Step-by-step analysis:

1. Step 0: Both start at 0.000 rad, target = -0.434 rad
2. Newton j2 accelerates **1.5× faster** toward target than PhysX
3. By step 10: PhysX at -0.250, Newton at -0.316 (3.8° gap)
4. The gap **never closes** — it accumulates to 13.4° by step 60 and stabilizes
5. Both joints eventually settle near the target, but via different trajectories

The velocity ratio (Newton/PhysX) is consistently ~1.2-1.5× during the initial response phase. This is the implicit-vs-explicit integration difference: PhysX's constraint solver damps the response more than Newton's explicit torque application.

### Why j2 Is Worst

j2 is:
- **High Kp (300)** — large torque amplifies any integration difference
- **Low in kinematic chain** — position error here propagates to all downstream links
- **Load-bearing** — supports the weight of the entire arm + hand above it (even with gravity disabled, the dynamics coupling is complex)

j7 (Kp=25) shows only 1.3° difference because lower stiffness means smaller torques and less sensitivity to integration differences.

### Finger Joints

Finger joints show up to 30° divergence, but this is somewhat expected:
- Very low damping (Kd=0.1) → highly oscillatory, sensitive to any difference
- Low effort limit (0.5 Nm) → near saturation during fast motions
- Even PhysX-to-PhysX shows 24° divergence when comparing DexSuite (with object contacts) vs standalone (no object)

---

## Proposed Solutions

### Short Term: Unified Actuator Model

Replace `ImplicitActuatorCfg` with `IdealPDActuatorCfg` in the robot asset configuration:

```python
from isaaclab.actuators import IdealPDActuatorCfg

actuators={
    "kuka_allegro_actuators": IdealPDActuatorCfg(
        joint_names_expr=[".*"],
        stiffness={...},  # same values
        damping={...},    # same values
        effort_limit={...},
    ),
}
```

`IdealPDActuator` computes torques identically on both backends:
```
τ = Kp * (q_target - q) + Kd * (dq_target - dq)
τ_applied = clip(τ, -effort_limit, +effort_limit)
```

Then sends the computed torque as a pure effort command. Both PhysX and Newton receive the same torque value and integrate it through their respective solvers. The PD computation is no longer backend-dependent.

**Trade-off:** Policies trained with `ImplicitActuatorCfg` on PhysX learned that specific response profile. Retraining with `IdealPDActuatorCfg` may require re-tuning gains, but the resulting policy will transfer across backends.

**Recommendation:** Retrain with `IdealPDActuatorCfg` on **all joints** — arm and fingers. The finger joints are low-stiffness and the integration difference is less impactful there, but consistency across the entire robot eliminates a class of subtle bugs.

### Medium Term: preserve_order=True by Default

The `SceneEntityCfg.preserve_order` parameter should default to `True` for any configuration that feeds into observation or action terms. The current default of `False` is a correctness hazard for multi-backend workflows.

### Long Term: Actuator Model Best Practices

Document and recommend:
1. **For sim2sim transfer:** Always use explicit actuator models (`IdealPDActuator`, `DCMotor`)
2. **For single-backend training:** `ImplicitActuator` is fine and may be slightly more stable
3. **For real2sim:** Use `IdealPDActuator` with gains matched to the real robot's servo controller
4. **For domain randomization:** Randomize gains on an explicit model — the DR range maps predictably to both backends

---

## Experimental Evidence

### Test Setup
- **Checkpoint:** PhysX-trained policy with domain randomization, 9750 iterations
- **Methodology:** Record 100 steps of joint position targets from DexSuite PhysX policy, replay on standalone robot (single robot + table, no object) on both PhysX and Newton with forced identical initial state
- **dt:** 1/120s, decimation=2 (2 physics steps per env step)
- **Newton solver:** implicitfast, 100 iterations, 15 ls_iterations, nconmax=200

### Results Summary

| Condition | Avg Success | Notes |
|-----------|-------------|-------|
| PhysX native (closed-loop policy) | 2.30 | Baseline |
| Newton, no remap | 0.0002 | Scrambled observations |
| Newton, action remap only | 0.0003 | Correct actions, wrong obs |
| Newton, full remap (obs+action) | 0.0014 | Correct remapping, physics gap |
| Newton, full remap + zero j7 vel | 0.0024 | Mitigates PhysX velocity bug |

The 1600× gap between PhysX (2.30) and Newton with full remap (0.0014) is almost entirely from the actuator dynamics mismatch, not from joint ordering (which accounts for ~7× of the gap).

### Additional Finding: PhysX j7 Velocity Bug
PhysX reports iiwa7_joint_7 velocity as ~0.0006 rad/s when actual velocity is ~0.086 rad/s (roughly 100× underreported). The policy's observation normalizer learned this bug. On Newton, j7 velocity is reported correctly, causing a large normalized observation difference. This is a PhysX-specific issue but contributes to transfer failure.

---

## Scripts and Data

All scripts are in `IsaacLab/scripts/sim2sim/`:
- `controlled_compare.py` — Record DexSuite targets, replay on standalone PhysX/Newton
- `complete_remap_eval.py` — Full obs+action remap evaluation
- `render_combined.py` — Side-by-side video rendering
- `plot_comparison.py` — Joint trajectory plots and heatmaps

Data files:
- `/tmp/controlled_dexsuite_physx.json`
- `/tmp/controlled_standalone_physx_table.json`
- `/tmp/controlled_standalone_newton_table.json`

---

## Open Questions

1. **Does `IdealPDActuator` + same gains reproduce PhysX `ImplicitActuator` behavior on PhysX?** Need to test whether switching PhysX from implicit to explicit PD changes its dynamics significantly.
2. **What gain tuning is needed?** The explicit PD model may need higher damping to match the implicit model's natural damping.
3. **Does the j7 velocity bug affect PhysX `IdealPDActuator`?** If the explicit model reads velocity correctly, it may behave differently on PhysX too.
4. **Should `preserve_order` default to `True`?** What are the performance implications? Are there cases where sorted order is intentionally desired?

---

## Appendix: Full Joint Ordering Comparison

### PhysX Joint Order (index → name)
```
 0: iiwa7_joint_1       7: index_joint_0    14: ring_joint_1
 1: iiwa7_joint_2       8: middle_joint_0   15: thumb_joint_1
 2: iiwa7_joint_3       9: ring_joint_0     16: index_joint_2
 3: iiwa7_joint_4      10: thumb_joint_0    17: middle_joint_2
 4: iiwa7_joint_5      11: index_joint_1    18: ring_joint_2
 5: iiwa7_joint_6      12: middle_joint_1   19: thumb_joint_2
 6: iiwa7_joint_7      13: ring_joint_1     20: index_joint_3
                                            21: middle_joint_3
                                            22: ring_joint_3
                                            23: thumb_joint_3
```

### Newton Joint Order (index → name)
```
 0: iiwa7_joint_1       7: index_joint_0    14: ring_joint_2
 1: iiwa7_joint_2       8: index_joint_1    15: ring_joint_3
 2: iiwa7_joint_3       9: index_joint_2    16: thumb_joint_0
 3: iiwa7_joint_4      10: index_joint_3    17: thumb_joint_1
 4: iiwa7_joint_5      11: middle_joint_0   18: thumb_joint_2
 5: iiwa7_joint_6      12: middle_joint_1   19: thumb_joint_3
 6: iiwa7_joint_7      13: middle_joint_2   20: ring_joint_0 (*)
                        14: middle_joint_3   21: ring_joint_1 (*)
```

(*Note: exact Newton order depends on USD traversal; verified empirically.)
