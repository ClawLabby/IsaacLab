# Sim2Sim Analysis: PhysX vs Newton Step Response Comparison

**Date**: 2026-04-04
**Scripts**: `scripts/sim2sim/solver_sweep.py`, `scripts/sim2sim/fixed_state_compare.py`

## Environment Solver Iteration Comparison

| Environment | Robot | solver_position_iterations |
|-------------|-------|--------------------------|
| DextrAH (sim2real proven) | Kuka-Allegro | **8** |
| DexSuite (Isaac Lab) | Kuka-Allegro | **32** |
| Allegro standalone | Allegro | 8 |
| Shadow Hand | Shadow Hand | 8 |
| Franka | Franka | 8 |
| ANYmal/Cassie/Spot | Quadruped | 4 |

DexSuite uses 4× more iterations than DextrAH and all other Isaac Lab robots.
However, **solver iterations only affect TGS contact solving, not PD articulation drives**.

## Step Response: Fixed Initial State, Action=1.0 Pulse (10 steps)

| Joint | Stiffness | Damping | Friction | PhysX Δ | Newton Δ | Ratio |
|-------|-----------|---------|----------|---------|----------|-------|
| iiwa7_joint_1 | 300 | 45 | 1.0 | 0.0838 | 0.0824 | 1.02× |
| iiwa7_joint_4 | 120 | 30 | 1.0 | 0.0900 | 0.0861 | 1.05× |
| iiwa7_joint_5 | 100 | 20 | 1.0 | 0.0751 | 0.0673 | 1.12× |
| iiwa7_joint_6 | 50 | 15 | 1.0 | 0.0492 | 0.0391 | 1.26× |
| iiwa7_joint_7 | 25 | 15 | 1.0 | 0.0143 | 0.0161 | 0.89× |
| index_joint_0 | 3 | 0.1 | 0.01 | 0.2302 | 0.2345 | 0.98× |
| thumb_joint_0 | 3 | 0.1 | 0.01 | 0.0708 | 0.0468 | 1.51× |

## Key Findings

### 1. PhysX Velocity Reporting Bug for iiwa7_joint_7
PhysX reports velocity ≈ 0.0006 rad/s for j7 while position clearly changes at 0.086 rad/s
(implied from Δpos/Δt). Newton correctly reports ~0.098 rad/s. This is NOT a physics
discrepancy — it's a velocity readback artifact in PhysX.

**Impact**: If the policy observes `joint_vel`, PhysX-trained policies learn with incorrect
velocity signal for j7, which won't transfer to Newton (where velocity is correct).

### 2. Actual Physics Within 10-26% for Mid-Range Joints
Position-based displacement ratios are 0.89× to 1.26× — reasonable for inter-solver comparison.
Not a fundamental barrier to policy transfer.

### 3. Joint Limit Enforcement Differs
PhysX: hard stop at limit, zero overshoot
Newton: overshoots then oscillates back (~25ms settling time)
The thumb_joint_0 shows 1.51× difference because PhysX stops at 1.5708 while Newton
reaches only 1.547 during the pulse due to different limit approach dynamics.

### 4. Solver Iterations Don't Affect PD Drives
Tested 4, 8, 16, 32 iterations — identical PD response. TGS iterations only matter for
rigid body contact resolution.

## Recommendations
1. **File PhysX bug**: velocity reporting for j7 (stiffness=25, damping=15, friction=1.0)
2. **Check velocity observations**: policies using joint_vel for j7 are affected
3. **Joint limit handling**: consider adding DR on joint limit enforcement for transfer
4. **DexSuite iteration count**: 32 is unnecessarily high vs DextrAH's 8 (which achieves sim2real)
