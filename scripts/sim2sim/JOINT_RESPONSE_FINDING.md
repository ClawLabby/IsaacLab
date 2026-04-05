# Sim2Sim Joint Response Analysis — Significant Finding

**Date**: 2026-04-05  
**Tool**: `scripts/sim2sim/joint_response.py`

## Test Configuration
- Robot: kuka_allegro
- Control: position target (δ = 0.1 rad during pulse)
- Gravity: OFF (zero-g)
- Newton: 2 substeps, MJWarp solver (iterations=100, ls_iterations=15)
- PhysX: default TGS solver
- dt = 1/120 s

## Result: 16× Response Difference

For `iiwa7_joint_1` (K=300, D=45):

| Step | PhysX pos | Newton pos | Pulse |
|------|-----------|------------|-------|
| 10   | 0.002     | -0.015     | Y     |
| 15   | 0.018     | 0.296      | Y     |
| 20   | 0.034     | 0.604      | Y     |
| 29   | 0.060     | 0.991      | Y     |

**Total displacement during pulse:**
- PhysX: 0.060 rad
- Newton: 0.991 rad  
- **Ratio: 16.4×**

## Observations

1. **Newton accelerates continuously** — velocity increases throughout pulse and beyond
2. **PhysX reaches quasi-equilibrium** — velocity stays ~0.3 rad/s, position creeps slowly
3. **Newton overshoots massively** — position exceeds target by 10× the intended delta
4. **Newton doesn't settle after pulse ends** — continues accelerating

## Hypothesis

The implicit actuator PD computation differs fundamentally:

**PhysX implicit PD**: Uses implicit integration where position target creates a virtual spring
that's integrated with stability constraints. The solver effectively limits how fast the joint
can respond per timestep.

**Newton MJWarp PD**: May be computing spring torque explicitly: τ = K × (target - pos) - D × vel.
With K=300 and target offset=0.1, this gives τ=30Nm initially. The explicit solver applies this
torque directly, causing large acceleration.

The DexSuite training uses startup domain randomization on stiffness (0.5-2×) and damping. 
Policies trained on PhysX learn to command aggressive position targets because the response
is damped. When transferred to Newton, those same commands cause wild overshoots.

## Next Steps

1. Verify by comparing applied torques between backends
2. Test with higher damping in Newton
3. Check if DextrAH uses different actuator config
4. Consider whether explicit torque control (not position target) gives better transfer
