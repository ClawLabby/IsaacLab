# Sim2Sim Transfer: Design Proposals for Isaac Lab

## Problem Statement

Users training RL policies on one physics backend (PhysX or Newton) currently get
silent failures when evaluating on another backend. The three root causes are:

1. **Joint ordering differs between backends** (breadth-first vs depth-first)
2. **`ImplicitActuatorCfg` produces different dynamics** (implicit vs explicit PD)
3. **No built-in tooling to detect or diagnose these issues**

## Proposal 1: Sim2Sim Validation Tool

A command-line tool that takes a trained checkpoint + task config and automatically
runs a cross-backend comparison:

```bash
isaaclab sim2sim-check \
  --task Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
  --checkpoint model_9750.pt \
  --source-backend physx \
  --target-backend newton \
  --num-episodes 10
```

**What it does:**
1. Runs the policy on the source backend, records joint targets + states
2. Replays the same targets on a standalone robot on both backends
3. Reports per-joint divergence with actionable diagnostics:
   - "Joint ordering mismatch detected: hand joints 7-22 are permuted"
   - "iiwa7_joint_2: 13.4° max divergence (Kp=300, implicit PD → explicit PD gap)"
   - "Recommend: switch to IdealPDActuatorCfg or add DR on Kp/Kd ×0.5-2.0"
4. Generates a visual comparison report (plots + optional video)

**Implementation:** Mostly exists already in our `scripts/sim2sim/` directory.
Needs packaging and integration with Isaac Lab CLI.

## Proposal 2: Backend-Portable Actuator Configurations

### Option A: `PortablePDActuatorCfg` (New Actuator Type)

A new actuator model that guarantees identical behavior on both backends:
- Computes PD torque explicitly: `τ = Kp*(target - pos) - Kd*vel`
- Applies torque as effort command (not position target)
- Same code path on PhysX and Newton
- Essentially `IdealPDActuator` but with automatic gain scaling to match
  the response profile of `ImplicitActuator` at the configured dt

The gain scaling is key: users specify "I want Kp=300 behavior like PhysX implicit"
and the actuator internally adjusts gains for the explicit computation to produce
a similar response curve.

### Option B: Unified Implicit PD Across Backends

Make Newton's implicit actuator implementation match PhysX's more closely.
This is the deeper fix but harder to implement — requires changes to MJWarp's
actuator integration scheme.

### Option C: Robot Asset Presets for Transfer

Add per-robot configuration presets that are validated for cross-backend transfer:

```python
KUKA_ALLEGRO_CFG_PORTABLE = KUKA_ALLEGRO_CFG.replace(
    actuators={
        "kuka_allegro_actuators": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={...},  # re-tuned gains for explicit PD
            damping={...},    # higher damping to compensate for lack of implicit damping
            effort_limit={...},
        ),
    },
)
```

Each portable config comes with:
- Validated gains that produce similar dynamics to the original
- Cross-backend test results showing expected divergence
- Documentation on the tradeoffs

## Proposal 3: Joint Ordering Safety Net

### Immediate Fix: Warning on Ordering Mismatch

When a policy trained on backend A is loaded on backend B, check if joint ordering
differs. If it does, emit a clear warning:

```
WARNING: Joint ordering differs between training backend (PhysX) and current backend (Newton).
  PhysX: index_joint_0, middle_joint_0, ring_joint_0, thumb_joint_0, ...
  Newton: index_joint_0, index_joint_1, index_joint_2, index_joint_3, ...
  
  Observations and actions will be permuted. Set preserve_order=True on your
  SceneEntityCfg or use the joint_remap utility to fix this.
```

### Medium-Term Fix: preserve_order=True Default

Change `SceneEntityCfg.preserve_order` default to `True` for new projects.
Existing projects can opt out with `preserve_order=False`.

### Long-Term Fix: Canonical Joint Ordering

Define a canonical ordering in the USD asset itself and enforce it across backends.
Both PhysX and Newton should enumerate joints in the same order as defined in the USD.

## Proposal 4: Intermediate Controller Layer

The DextrAH approach of using geometric fabrics as an intermediate controller
between the policy and the actuators provides natural sim2real/sim2sim robustness.
Proposal: make fabric-like controllers first-class in Isaac Lab.

**Practical version:** A `SmoothedPositionActionCfg` that applies exponential
smoothing or rate limiting to position targets before sending them to the actuator:

```python
class SmoothedPositionActionCfg(ActionTermCfg):
    """Position action with configurable smoothing/rate limiting.
    
    Filters raw policy outputs to produce smooth joint targets,
    reducing sensitivity to actuator model differences.
    """
    max_delta_per_step: float = 0.1  # rad — max change per env step
    smoothing_alpha: float = 0.3     # exponential smoothing factor
    scale: float = 0.1              # action scaling (same as RelativeJointPositionAction)
```

This doesn't require the full geometric fabric machinery but captures the key
benefit: the actuator receives smooth, physically realizable targets instead of
janky policy outputs.

## Proposal 5: Actuator Model Documentation Guide

For end users going from robot spec sheets to sim configuration:

### Decision Tree:

1. **What is your real robot's control interface?**
   - Position commands to firmware servo → `ImplicitActuatorCfg`
   - Position commands to software PD (e.g., Allegro SDK) → `IdealPDActuatorCfg`
   - Direct torque commands → `IdealPDActuatorCfg` or custom effort model
   - Task-space commands via controller → Use fabric/OSC action space

2. **Will you transfer between physics backends?**
   - Yes → Use `IdealPDActuatorCfg` with re-tuned gains
   - No → `ImplicitActuatorCfg` is fine and may be more stable

3. **What Kp/Kd values to use?**
   - Start from manufacturer spec sheet or system identification
   - For `ImplicitActuatorCfg`: gains can be used directly
   - For `IdealPDActuatorCfg`: multiply Kd by ~1.5-2x to compensate for 
     lack of implicit numerical damping
   - Always validate with step response test on single joint

4. **Domain randomization range?**
   - Kp: ×0.5 to ×2.0 (covers manufacturing variation + model uncertainty)
   - Kd: ×0.5 to ×2.0
   - Joint friction: 0 to 5× nominal
   - If targeting sim2sim: DR should span the implicit↔explicit PD gap (~1.5x velocity response difference)

### Per-Robot Templates:

Provide tested configurations for common robots:
- Kuka iiwa + Allegro (DexSuite/DextrAH)
- Franka Panda
- UR5/UR10
- Unitree H1/G1
- Shadow Hand

Each template includes:
- Recommended actuator model and gains
- Expected cross-backend divergence
- DR ranges for sim2real
- Known issues and workarounds

## Priority Ranking

1. **Joint ordering warning** (Proposal 3, immediate fix) — low effort, prevents silent failures
2. **Sim2Sim validation tool** (Proposal 1) — packages existing work, high user value
3. **Actuator documentation guide** (Proposal 5) — directly answers user needs
4. **Smoothed position action** (Proposal 4) — moderate effort, high impact for transfer
5. **Backend-portable actuator configs** (Proposal 2C) — requires gain tuning validation
6. **preserve_order default change** (Proposal 3, medium-term) — API change, needs migration path
7. **Unified implicit PD** (Proposal 2B) — hard, long-term, ideal solution
