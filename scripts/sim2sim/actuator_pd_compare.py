"""Compare Newton joint response: built-in MuJoCo actuator vs external ActuatorPD.

This test demonstrates the difference between:
1. Newton's built-in PD (target_ke/target_kd → MuJoCo actuator gains)
2. External ActuatorPD from newton-actuators (zero internal gains, explicit torque to joint_f)

Both should produce identical torques for the same state, but the integration
may differ because MuJoCo treats internal gains vs external forces differently.

Usage:
  cd /home/horde/claw/git/IsaacLab
  PYTHONDONTWRITEBYTECODE=1 env_isaaclab/bin/python scripts/sim2sim/actuator_pd_compare.py
"""

import os
import sys
import math
import argparse

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=120)
parser.add_argument("--pulse_start", type=int, default=10)
parser.add_argument("--pulse_end", type=int, default=50)
parser.add_argument("--target_delta", type=float, default=0.1)
args = parser.parse_args()

import numpy as np
import warp as wp

wp.init()

import newton
from newton_actuators import ActuatorPD

# DexSuite joint parameters
JOINTS = [
    ("iiwa7_joint_1", 300.0, 45.0, 300.0),
    ("iiwa7_joint_6", 50.0,  15.0, 50.0),
    ("iiwa7_joint_7", 25.0,  15.0, 25.0),
    ("index_joint_0", 3.0,   0.1,  0.7),
    ("thumb_joint_0", 3.0,   0.1,  0.7),
]

device = "cuda:0"
dt = 1.0 / 120.0


def build_model(kp, kd, max_f, use_internal_pd=True):
    """Build a single revolute joint model."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    mass = 1.0
    inertia = wp.mat33((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1))

    b = builder.add_link(armature=0.0, inertia=inertia, mass=mass)
    builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1,
                          cfg=newton.ModelBuilder.ShapeConfig(density=1))

    if use_internal_pd:
        # Built-in MuJoCo actuator gains
        target_ke = kp
        target_kd = kd
    else:
        # Zero internal gains — we apply forces externally
        target_ke = 0.0
        target_kd = 0.0

    j = builder.add_joint_revolute(
        parent=-1,
        child=b,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_pos=0.0,
        target_vel=0.0,
        target_ke=target_ke,
        target_kd=target_kd,
        armature=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )
    builder.add_articulation([j])

    return builder.finalize(device=device)


def run_builtin_pd(name, kp, kd, max_f):
    """Run with Newton's built-in MuJoCo actuator."""
    model = build_model(kp, kd, max_f, use_internal_pd=True)
    solver = newton.solvers.SolverMuJoCo(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    positions = []
    velocities = []

    for step in range(args.num_steps):
        target = args.target_delta if args.pulse_start <= step < args.pulse_end else 0.0

        control.joint_target_pos = wp.array([target], dtype=wp.float32, device=device)
        control.joint_target_vel = wp.array([0.0], dtype=wp.float32, device=device)

        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

        if not isinstance(solver, newton.solvers.SolverFeatherstone):
            newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

        positions.append(float(state_0.joint_q.numpy()[0]))
        velocities.append(float(state_0.joint_qd.numpy()[0]))

    return np.array(positions), np.array(velocities)


def run_external_actuator_pd(name, kp, kd, max_f):
    """Run with newton-actuators ActuatorPD (zero internal gains)."""
    model = build_model(kp, kd, max_f, use_internal_pd=False)
    solver = newton.solvers.SolverMuJoCo(model)

    # Create external ActuatorPD
    input_idx = wp.array([0], dtype=wp.uint32, device=device)
    output_idx = wp.array([0], dtype=wp.uint32, device=device)
    kp_arr = wp.array([kp], dtype=wp.float32, device=device)
    kd_arr = wp.array([kd], dtype=wp.float32, device=device)
    max_f_arr = wp.array([max_f], dtype=wp.float32, device=device)
    gear_arr = wp.array([1.0], dtype=wp.float32, device=device)

    actuator = ActuatorPD(
        input_indices=input_idx,
        output_indices=output_idx,
        kp=kp_arr,
        kd=kd_arr,
        max_force=max_f_arr,
        gear=gear_arr,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    positions = []
    velocities = []
    applied_torques = []

    for step in range(args.num_steps):
        target = args.target_delta if args.pulse_start <= step < args.pulse_end else 0.0

        control.joint_target_pos = wp.array([target], dtype=wp.float32, device=device)
        control.joint_target_vel = wp.array([0.0], dtype=wp.float32, device=device)

        state_0.clear_forces()
        control.joint_f.zero_()

        # Apply external ActuatorPD
        actuator.step(state_0, control, dt=dt)

        torque_val = float(control.joint_f.numpy()[0])
        applied_torques.append(torque_val)

        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

        if not isinstance(solver, newton.solvers.SolverFeatherstone):
            newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

        positions.append(float(state_0.joint_q.numpy()[0]))
        velocities.append(float(state_0.joint_qd.numpy()[0]))

    return np.array(positions), np.array(velocities), np.array(applied_torques)


def main():
    print("=" * 80)
    print("Newton Joint Response: Built-in MuJoCo PD vs External ActuatorPD")
    print("=" * 80)
    print(f"Steps: {args.num_steps}, dt=1/120, pulse: steps {args.pulse_start}-{args.pulse_end}")
    print(f"Target delta: {args.target_delta} rad")
    print()

    for name, kp, kd, max_f in JOINTS:
        print(f"\n{'─' * 70}")
        print(f"Joint: {name} (Kp={kp}, Kd={kd}, max_force={max_f})")
        print(f"{'─' * 70}")

        # Run both
        builtin_pos, builtin_vel = run_builtin_pd(name, kp, kd, max_f)
        ext_pos, ext_vel, ext_torque = run_external_actuator_pd(name, kp, kd, max_f)

        # Compare at pulse end
        pe = min(args.pulse_end, len(builtin_pos)) - 1
        b_disp = builtin_pos[pe]
        e_disp = ext_pos[pe]
        ratio = abs(b_disp / e_disp) if abs(e_disp) > 1e-8 else float('inf')

        print(f"\nDisplacement at step {pe}:")
        print(f"  Built-in MuJoCo PD: {b_disp:+.6f} rad")
        print(f"  External ActuatorPD: {e_disp:+.6f} rad")
        print(f"  Ratio (builtin/ext): {ratio:.4f}×")

        # Overall trajectory difference
        pos_diff = np.abs(builtin_pos - ext_pos)
        vel_diff = np.abs(builtin_vel - ext_vel)
        print(f"\nTrajectory difference:")
        print(f"  Position — max: {pos_diff.max():.6f} rad, mean: {pos_diff.mean():.6f} rad")
        print(f"  Velocity — max: {vel_diff.max():.6f} rad/s, mean: {vel_diff.mean():.6f} rad/s")

        # Step-by-step during pulse start
        n_show = min(15, args.pulse_end - args.pulse_start)
        print(f"\n{'Step':>5} {'Builtin pos':>13} {'ExtPD pos':>13} {'ExtPD torque':>13} {'Pos diff':>10}")
        for s in range(args.pulse_start, args.pulse_start + n_show):
            diff = abs(builtin_pos[s] - ext_pos[s])
            print(f"{s:5d} {builtin_pos[s]:+13.6f} {ext_pos[s]:+13.6f} {ext_torque[s]:+13.4f} {diff:10.6f}")

    print(f"\n{'=' * 80}")
    print("INTERPRETATION:")
    print("  If built-in ≈ external: MuJoCo treats internal gains and joint_f identically.")
    print("  If built-in ≠ external: MuJoCo integrates internal gains differently (implicit")
    print("    vs explicit), and using ActuatorPD changes Newton's behavior.")
    print("=" * 80)


main()
