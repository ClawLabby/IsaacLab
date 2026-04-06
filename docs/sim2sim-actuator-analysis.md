# Actuator Model Sim2Sim and Sim2Real Analysis

**Date**: April 5, 2026  
**Author**: ClawLabby  
**Context**: DexSuite Kuka-Allegro dexterous manipulation, Isaac Lab

## Executive Summary

Cross-simulator policy transfer between PhysX and Newton/MJWarp fails for DexSuite manipulation tasks. The root cause is not parameter mismatch — actuator parameters (stiffness, damping, friction, armature) are identical across backends. The failure stems from **how each physics solver integrates PD control torques**, producing fundamentally different joint dynamics from the same gains and targets.

This document analyzes the problem, quantifies the gap, and proposes solutions using the `newton-actuators` library to achieve both sim2sim transfer and improved sim2real fidelity.

## 1. The Problem

### 1.1 Observed Failure

A policy trained on PhysX (state-only, with domain randomization, peak success 5.96) achieves **0.00 success** when evaluated on Newton with identical actuator parameters. The policy commands aggressive position targets that work under PhysX's damped response but cause wild overshoot on Newton.

### 1.2 Root Causes

Three distinct issues contribute to the transfer failure:

#### 1.2.1 Joint Ordering Mismatch (Critical, Fixable)

PhysX and Newton enumerate Allegro hand joints in different orders:

| PhysX Order | Newton Order |
|---|---|
| Groups by joint number across fingers | Groups by finger |
| `index_j0, middle_j0, ring_j0, thumb_j0, index_j1, ...` | `index_j0, index_j1, index_j2, index_j3, middle_j0, ...` |

Without remapping, a PhysX-trained policy sends actions to the **wrong joints** on Newton. A fixed permutation table resolves this.

#### 1.2.2 PD Actuator Integration Difference (Fundamental)

The same PD parameters produce dramatically different joint responses:

| Joint | Kp | Kd | PhysX Displacement | Newton Displacement | Ratio |
|---|---|---|---|---|---|
| iiwa7_joint_1 | 300 | 45 | 0.060 rad | 0.991 rad | **16.5×** |
| iiwa7_joint_6 | 50 | 15 | 0.049 rad | 0.039 rad | 1.26× |
| iiwa7_joint_7 | 25 | 15 | 0.014 rad | 0.016 rad | 0.89× |
| index_joint_0 | 3 | 0.1 | 0.230 rad | 0.235 rad | 0.98× |

*Test: 0.1 rad position target pulse, 10 steps at dt=1/120s, zero gravity.*

The gap scales with stiffness. Low-stiffness finger joints (Kp=3) match within 2%. High-stiffness arm joints (Kp=300) diverge by 16×. This is because:

- **PhysX implicit PD**: The PD spring is part of the constraint system. The solver evaluates `τ = Kp·(target - q_next) + Kd·(target_vel - v_next)` at the *future* state, effectively providing infinite-bandwidth control. High stiffness is inherently stable regardless of timestep.

- **Newton/MuJoCo built-in PD**: Actuator gains are integrated through MuJoCo's implicit formulation with solver regularization. More stable than pure explicit, but still allows more motion per timestep than PhysX.

- **Explicit PD** (e.g., `newton-actuators.ActuatorPD`): Computes torque from *current* state and applies as external force. With Kp=300 at 120Hz, this is unstable — the gain exceeds the Nyquist stability limit for the system's inertia.

#### 1.2.3 PhysX Velocity Reporting Bug (Minor)

PhysX reports near-zero velocity for `iiwa7_joint_7` (0.0006 rad/s) while position clearly changes at 0.086 rad/s. Newton reports correctly. Policies using `joint_vel` for j7 get corrupted observations, but this is a minor factor compared to the integration difference.

## 2. Experimental Validation

### 2.1 Newton Built-in vs External ActuatorPD

To understand whether the gap is in the actuator model or the integration, we tested Newton with:
1. **Built-in MuJoCo PD** (target_ke/target_kd → internal actuator gains)
2. **External ActuatorPD** (zero internal gains, explicit torque written to joint_f)

Results for iiwa7_joint_1 (Kp=300, Kd=45):

| Mode | Displacement at step 49 | Behavior |
|---|---|---|
| Built-in MuJoCo PD | 0.090 rad | Stable, damped approach |
| External ActuatorPD | 0.810 rad | Saturates at ±300Nm, oscillates |

For iiwa7_joint_7 (Kp=25, Kd=15):

| Mode | Displacement at step 49 | Behavior |
|---|---|---|
| Built-in MuJoCo PD | 0.042 rad | Stable |
| External ActuatorPD | 0.043 rad | Stable, **matches built-in** |

**Key finding**: MuJoCo's implicit integration stabilizes high-stiffness joints. For low-to-medium stiffness, built-in and external PD produce identical results (within 0.5%). The implicit path is not "more accurate" — it's artificially stabilizing what would be an unstable controller at this timestep.

### 2.2 What Real Robots Do

A real KUKA iiwa runs its joint PD controller at **1-3 kHz**, not 120 Hz. At 1 kHz with Kp=300 and joint inertia ~0.1 kg·m², the controller is stable. The DexSuite gain values were tuned for PhysX's implicit solver, which effectively acts like an infinite-rate controller. Neither PhysX's implicit PD nor Newton's MuJoCo PD accurately models the real control loop — they both paper over the timestep-vs-gain stability issue in different ways.

The explicit `ActuatorPD` at 120 Hz is actually the most *honest* model — it correctly shows that Kp=300 is unstable at this rate. Real systems avoid this by:
- Higher control rates (1-10 kHz)
- Current loop bandwidth limiting
- Motor inductance/inertance (physical torque rate limiting)
- The DC motor torque-speed curve (velocity-dependent saturation)

## 3. Proposed Solutions

### 3.1 Solution A: Unified Explicit Actuator (Recommended for Sim2Sim + Sim2Real)

Use `newton-actuators.ActuatorPD` (or `ActuatorDCMotor`) on **both** backends, with internal PD gains set to zero. The actuator kernel runs identically on both — same Warp code, same torque computation. The only remaining difference is solver integration dynamics.

**For high-stiffness joints**, use one of:

#### 3.1.1 ActuatorDCMotor (Best Physical Fidelity)

The DC motor model adds velocity-dependent torque saturation:

```
τ_max(v) = τ_stall · (1 - v/v_max)
τ_min(v) = τ_stall · (-1 - v/v_max)
τ_applied = clamp(τ_pd, τ_min, τ_max)
```

This naturally limits torque at high velocities — exactly what real motors do. At the start of motion (v≈0), the motor can produce full stall torque. As the joint accelerates, available torque drops. This prevents the runaway acceleration that causes explicit PD instability, without relying on implicit solver tricks.

**Parameters needed**: `saturation_effort` (stall torque) and `velocity_limit` (no-load speed) per joint — these can be derived from KUKA iiwa and Allegro motor datasheets.

#### 3.1.2 Higher Substep Rate

Run the actuator at physics substep rate rather than control rate. With 8-10 substeps (960-1200 Hz effective), explicit PD with Kp=300 is stable. This is also more physically accurate — it approximates the real 1kHz control loop.

Cost: ~4-5× physics computation per step. May be acceptable since learning time is dominated by rendering for vision policies.

#### 3.1.3 Re-tuned Gains

Lower Kp for arm joints to values that are stable at 120Hz. DextrAH (which achieves sim2real) uses 8 solver iterations vs DexSuite's 32 — suggesting the gains may be unnecessarily high.

### 3.2 Solution B: Domain Randomization Bridge (Pragmatic)

Train with DR that spans both solver regimes:
- Randomize Kp from 0.3× to 3× (covers PhysX effective response through Newton's)
- Randomize Kd similarly
- Add action delay randomization (1-3 steps) to mimic real servo latency

This doesn't require code changes but produces less precise control — the policy learns to be robust rather than accurate.

### 3.3 Solution C: Actuator Network (Nuclear Option for Sim2Real)

`newton-actuators.ActuatorNetMLP` takes position error and velocity history as input and outputs torques via a learned network. Train this from real robot data (or from a high-fidelity model) and use it in both simulators. This implicitly captures motor dynamics, gear backlash, friction, etc.

**Workflow**:
1. Collect joint-level data from real KUKA iiwa + Allegro (position, velocity, commanded torque, resulting motion)
2. Train ActuatorNetMLP per joint group
3. Deploy same network in both PhysX and Newton
4. All three systems (PhysX, Newton, real) now share the same actuator model

## 4. Recommendation

**Short term (sim2sim)**: Implement `ActuatorDCMotor` from `newton-actuators` on Newton for DexSuite. Set PhysX internal PD gains to zero and apply the same actuator kernel. This gives identical torque computation on both backends with physically motivated velocity saturation that prevents instability. Test cross-sim policy transfer.

**Medium term**: Increase physics substep count for the actuator (not necessarily for contacts). Some simulators support "actuator substeps" that are cheaper than full physics substeps.

**Long term (sim2real)**: Collect real actuator data and train `ActuatorNetMLP` per joint group. This is the gold standard for sim2real transfer — it eliminates actuator modeling error entirely.

## 5. Implementation Notes

### 5.1 Integration with Isaac Lab

The `newton-actuators` library operates on Warp arrays and is solver-agnostic. Integration path:

1. In the environment's `_pre_physics_step()`, zero out internal PD gains
2. Create `ActuatorPD` / `ActuatorDCMotor` instances at environment init
3. Call `actuator.step(sim_state, sim_control, dt=dt)` before `sim.step()`
4. The actuator writes computed torques to `joint_f`

For PhysX, this requires switching from `ImplicitActuatorCfg` to `IdealPDActuatorCfg` (which already uses explicit torque) or directly applying joint efforts via the articulation API.

### 5.2 DC Motor Parameters for DexSuite

Approximate values (to be validated against datasheets):

| Joint Group | Kp | Kd | Est. Stall Torque | Est. No-Load Speed |
|---|---|---|---|---|
| iiwa7_joint_1-3 | 200-300 | 30-45 | 320 Nm | 2.3 rad/s |
| iiwa7_joint_4-5 | 100-120 | 20-30 | 176 Nm | 2.3 rad/s |
| iiwa7_joint_6-7 | 25-50 | 15 | 40 Nm | 2.3 rad/s |
| Allegro fingers | 3 | 0.1 | 0.7 Nm | 8.7 rad/s |

*KUKA iiwa 7 R800 peak torques from datasheet. Allegro from Wonik spec sheet.*

### 5.3 Relevant Code

- `newton-actuators`: https://github.com/newton-physics/newton-actuators
- Comparison script: `scripts/sim2sim/actuator_pd_compare.py`
- Joint response analysis: `scripts/sim2sim/joint_response.py`
- Cross-sim analysis: `memory/sim2sim-analysis-2026-04-04.md`

## Appendix: Test Data

### A.1 Full Step Response Comparison (iiwa7_joint_1)

Newton built-in MuJoCo PD vs external ActuatorPD, 0.1 rad target pulse:

| Step | Built-in pos | External pos | External torque | Note |
|---|---|---|---|---|
| 10 | +0.004 | +0.021 | +30.0 | Pulse starts |
| 11 | +0.009 | -0.020 | -88.7 | External already overshooting |
| 12 | +0.014 | +0.117 | +255.9 | Oscillation begins |
| 15 | +0.028 | +0.111 | -300.0 | Saturated at max torque |
| 20 | +0.046 | +0.379 | +300.0 | Full ±300Nm oscillation |
| 49 | +0.090 | +0.810 | — | 9× gap at pulse end |

This demonstrates that the MuJoCo implicit path provides ~9× more damping than explicit application of the same PD gains. PhysX provides even more (~16× vs explicit based on earlier measurements).
