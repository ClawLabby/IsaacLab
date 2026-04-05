#!/usr/bin/env python3
"""Zero-gravity joint response comparison tool.

Spawns a robot in free space (no gravity, no ground, no objects) and applies
controlled effort inputs to each joint individually. Measures position and
velocity response to isolate articulation solver behavior from contact/gravity
confounds.

Usage:
    # Run on both backends, then compare:
    python joint_response.py --robot kuka_allegro --backend physx --output /tmp/joint_response
    python joint_response.py --robot kuka_allegro --backend newton --output /tmp/joint_response
    python joint_response.py --compare /tmp/joint_response --robot kuka_allegro

    # Or run both + compare automatically:
    python joint_response.py --robot kuka_allegro --output /tmp/joint_response --both

Supported robots: kuka_allegro, allegro, shadow_hand, franka
"""

import argparse
import json
import os
import sys

# ── Argument parsing (before Isaac Lab import) ───────────────────────────────

parser = argparse.ArgumentParser(description="Zero-gravity joint response comparison")
parser.add_argument("--robot", type=str, default="kuka_allegro",
                    choices=["kuka_allegro", "allegro", "shadow_hand", "franka"],
                    help="Robot to test")
parser.add_argument("--backend", type=str, default=None, choices=["physx", "newton"],
                    help="Physics backend")
parser.add_argument("--output", type=str, default="/tmp/joint_response",
                    help="Output directory for JSON results")
parser.add_argument("--compare", type=str, default=None,
                    help="Compare-only mode: directory containing both result JSONs")
parser.add_argument("--both", action="store_true",
                    help="Run both backends sequentially, then compare")
parser.add_argument("--steps", type=int, default=60,
                    help="Total steps per joint test (default: 60)")
parser.add_argument("--pulse-start", type=int, default=10,
                    help="Step to begin effort pulse (default: 10)")
parser.add_argument("--pulse-end", type=int, default=30,
                    help="Step to end effort pulse (default: 30)")
parser.add_argument("--effort", type=float, default=1.0,
                    help="Effort magnitude during pulse in Nm (default: 1.0)")
parser.add_argument("--joints", type=str, default=None,
                    help="Comma-separated joint names to test (default: all)")
parser.add_argument("--solver-iters", type=int, default=None,
                    help="Override solver_position_iteration_count (PhysX only)")

args, remaining = parser.parse_known_args()

# ═══════════════════════════════════════════════════════════════════════════════
# Compare mode — no simulation needed
# ═══════════════════════════════════════════════════════════════════════════════

def run_compare(directory: str, robot: str):
    """Load PhysX + Newton JSONs, print side-by-side analysis."""
    import numpy as np

    physx_file = os.path.join(directory, f"{robot}_physx.json")
    newton_file = os.path.join(directory, f"{robot}_newton.json")

    for f in [physx_file, newton_file]:
        if not os.path.exists(f):
            print(f"Missing: {f}")
            sys.exit(1)

    p = json.load(open(physx_file))
    n = json.load(open(newton_file))

    p_names = p["joint_names"]
    n_idx = {name: i for i, name in enumerate(n["joint_names"])}

    ps, pe = p["pulse_start"], p["pulse_end"]
    mid = (ps + pe) // 2

    W = 95
    print("=" * W)
    print(f"  JOINT RESPONSE COMPARISON: {robot} — PhysX vs Newton")
    print(f"  Zero gravity | effort={p['effort']} Nm | pulse steps {ps}–{pe} | dt={p['sim_dt']}")
    print("=" * W)

    print(f"\n{'Joint':<20} {'K':>5} {'D':>5} {'F':>5} | "
          f"{'P_Δpos':>9} {'N_Δpos':>9} {'Ratio':>6} | "
          f"{'P_ivel':>8} {'N_ivel':>8} {'VRat':>6} | Notes")
    print("-" * W)

    ratios, vel_bugs = [], []

    for j_name in p_names:
        if j_name not in p["tests"] or j_name not in n["tests"]:
            continue

        pt = p["tests"][j_name]["trajectory"]
        nt = n["tests"][j_name]["trajectory"]
        params = p["tests"][j_name]["params"]
        K = params.get("stiffness", 0)
        D = params.get("damping", 0)
        F = params.get("friction", 0)

        # Displacement during pulse
        p_disp = abs(pt[pe - 1]["pos"] - pt[ps - 1]["pos"])
        n_disp = abs(nt[pe - 1]["pos"] - nt[ps - 1]["pos"])
        ratio = p_disp / n_disp if n_disp > 1e-8 else float('inf')
        ratios.append(ratio)

        # Mid-pulse implied velocity
        p_iv = pt[mid]["implied_vel"]
        n_iv = nt[mid]["implied_vel"]
        vr = p_iv / n_iv if abs(n_iv) > 1e-8 else float('inf')

        # Velocity reporting check
        p_rv = pt[mid]["vel"]
        vel_ok = abs(p_rv - p_iv) / (abs(p_iv) + 1e-8) < 0.5

        notes = []
        if not vel_ok:
            notes.append("VEL_BUG")
            vel_bugs.append(j_name)
        if ratio > 1.5 or ratio < 0.67:
            notes.append("HIGH_DIFF")

        # Joint limit proximity
        lim = p["tests"][j_name]["limits"]
        end_pos = pt[pe - 1]["pos"]
        if lim[1] - lim[0] > 0 and (end_pos >= lim[1] * 0.98 or end_pos <= lim[0] * 0.98):
            notes.append("LIMIT")

        print(f"{j_name:<20} {K:>5.0f} {D:>5.1f} {F:>5.2f} | "
              f"{p_disp:>9.6f} {n_disp:>9.6f} {ratio:>6.3f} | "
              f"{p_iv:>8.4f} {n_iv:>8.4f} {vr:>6.3f} | {' '.join(notes)}")

    print()
    if ratios:
        print(f"  Displacement ratio: mean={np.mean(ratios):.3f}, "
              f"std={np.std(ratios):.3f}, range=[{min(ratios):.3f}, {max(ratios):.3f}]")
    if vel_bugs:
        print(f"  ⚠️  PhysX velocity reporting anomalies: {vel_bugs}")
    print(f"\n  Files: {physx_file}")
    print(f"         {newton_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch: --compare, --both, or single-backend simulation
# ═══════════════════════════════════════════════════════════════════════════════

if args.compare:
    run_compare(args.compare, args.robot)
    sys.exit(0)

if args.both:
    import subprocess
    base_cmd = [sys.executable, __file__,
                "--robot", args.robot, "--output", args.output,
                "--steps", str(args.steps),
                "--pulse-start", str(args.pulse_start),
                "--pulse-end", str(args.pulse_end),
                "--effort", str(args.effort)]
    if args.solver_iters is not None:
        base_cmd += ["--solver-iters", str(args.solver_iters)]
    if args.joints:
        base_cmd += ["--joints", args.joints]
    base_cmd += remaining

    for backend in ["physx", "newton"]:
        print(f"\n{'=' * 60}\n  Running {backend}...\n{'=' * 60}\n")
        result = subprocess.run(
            base_cmd + ["--backend", backend, "--headless"],
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "yes"})
        if result.returncode != 0:
            print(f"ERROR: {backend} failed (code {result.returncode})")
            sys.exit(1)

    print(f"\n{'=' * 60}\n  Comparing...\n{'=' * 60}\n")
    run_compare(args.output, args.robot)
    sys.exit(0)

if args.backend is None:
    parser.error("--backend is required (or use --compare / --both)")

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation mode — launch Isaac Sim
# ═══════════════════════════════════════════════════════════════════════════════

# Inject backend preset into sys.argv for Hydra/IsaacLab
sim_argv = [sys.argv[0]] + remaining
sys.argv = sim_argv

os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

from isaaclab.app import AppLauncher  # noqa: E402
launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = launcher.app

import torch  # noqa: E402
import warp as wp  # noqa: E402
import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

# Select physics backend
if args.backend == "newton":
    from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
    physics_cfg = NewtonCfg(
        num_substeps=2,  # Match DexSuite config
        solver_cfg=MJWarpSolverCfg(
            iterations=100,
            ls_iterations=15,
        ),
    )
    print(f"Physics backend: Newton (MJWarp solver, 2 substeps)")
else:
    from isaaclab_physx.physics import PhysxCfg
    physics_cfg = PhysxCfg()
    print(f"Physics backend: PhysX")


def to_np(x):
    """Robust tensor → numpy conversion."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    try:
        if isinstance(x, wp.array):
            return wp.to_torch(x).cpu().numpy()
    except Exception:
        pass
    return np.array(x)


def get_robot_cfg(name: str) -> ArticulationCfg:
    """Load robot config by name."""
    if name == "kuka_allegro":
        from isaaclab_assets import KUKA_ALLEGRO_CFG
        return KUKA_ALLEGRO_CFG
    elif name == "allegro":
        from isaaclab_assets import ALLEGRO_HAND_CFG
        return ALLEGRO_HAND_CFG
    elif name == "shadow_hand":
        from isaaclab_assets import SHADOW_HAND_CFG
        return SHADOW_HAND_CFG
    elif name == "franka":
        from isaaclab_assets import FRANKA_PANDA_CFG
        return FRANKA_PANDA_CFG
    raise ValueError(f"Unknown robot: {name}")


# ── Build scene config ────────────────────────────────────────────────────────

robot_cfg = get_robot_cfg(args.robot).replace(prim_path="{ENV_REGEX_NS}/Robot")

# Disable gravity on robot rigid bodies
if hasattr(robot_cfg.spawn, 'rigid_props') and robot_cfg.spawn.rigid_props is not None:
    robot_cfg.spawn.rigid_props.disable_gravity = True

# Override solver iterations
if args.solver_iters is not None and hasattr(robot_cfg.spawn, 'articulation_props'):
    if robot_cfg.spawn.articulation_props is not None:
        robot_cfg.spawn.articulation_props.solver_position_iteration_count = args.solver_iters


@configclass
class JointResponseSceneCfg(InteractiveSceneCfg):
    """Minimal scene: robot in empty space."""
    robot: ArticulationCfg = robot_cfg


# ── Simulation ────────────────────────────────────────────────────────────────

sim_cfg = sim_utils.SimulationCfg(
    device="cuda:0",
    dt=1.0 / 120.0,
    gravity=(0.0, 0.0, 0.0),
    physics=physics_cfg,
)
sim = SimulationContext(sim_cfg)
scene = InteractiveScene(JointResponseSceneCfg(num_envs=1, env_spacing=3.0))
sim.reset()

robot = scene["robot"]
joint_names = robot.joint_names
num_joints = len(joint_names)
sim_dt = sim.get_physics_dt()

print(f"\nBackend: {args.backend}")
print(f"Robot: {args.robot} ({num_joints} joints)")
print(f"dt: {sim_dt}, gravity: off")

# Which joints to test
if args.joints:
    test_names = [j.strip() for j in args.joints.split(",")]
else:
    test_names = list(joint_names)

# Gather actuator parameters
act_params = {}
for act_name, actuator in robot.actuators.items():
    stiff = to_np(actuator.stiffness).flatten()
    damp = to_np(actuator.damping).flatten()
    effort = to_np(actuator.effort_limit).flatten()
    fric = to_np(actuator.friction).flatten() if hasattr(actuator, 'friction') else np.zeros_like(stiff)
    arm = to_np(actuator.armature).flatten() if hasattr(actuator, 'armature') else np.zeros_like(stiff)

    ji = actuator.joint_indices
    if isinstance(ji, slice):
        indices = list(range(num_joints))
    else:
        indices = to_np(ji).flatten().astype(int).tolist()
    for local_i, global_i in enumerate(indices):
        act_params[joint_names[global_i]] = {
            "stiffness": float(stiff[local_i]),
            "damping": float(damp[local_i]),
            "effort_limit": float(effort[local_i]),
            "friction": float(fric[local_i]),
            "armature": float(arm[local_i]),
        }

# Joint limits
_lims = to_np(robot.data.soft_joint_pos_limits)  # shape (num_envs, num_joints, 2)
lim_lo = _lims[0, :, 0].flatten()
lim_hi = _lims[0, :, 1].flatten()

# Initial state
init_pos = wp.to_torch(robot.data.default_joint_pos).clone()
init_vel = torch.zeros_like(init_pos)

# ── Results structure ─────────────────────────────────────────────────────────

results = {
    "tool": "joint_response",
    "version": "1.1",
    "robot": args.robot,
    "backend": args.backend,
    "sim_dt": sim_dt,
    "gravity": [0, 0, 0],
    "control_mode": "position_target",
    "target_delta": 0.1,  # rad offset during pulse
    "num_steps": args.steps,
    "pulse_start": args.pulse_start,
    "pulse_end": args.pulse_end,
    "solver_iters_override": args.solver_iters,
    "joint_names": list(joint_names),
    "initial_positions": {name: float(to_np(init_pos).flatten()[i]) for i, name in enumerate(joint_names)},
    "actuator_params": act_params,
    "tests": {},
}

# ── Per-joint test loop ───────────────────────────────────────────────────────

for j_name in test_names:
    j_idx = joint_names.index(j_name)
    j_init = float(to_np(init_pos).flatten()[j_idx])
    params = act_params.get(j_name, {})

    print(f"\n  {j_name} (K={params.get('stiffness','?')}, D={params.get('damping','?')}, "
          f"F={params.get('friction','?')}, limits=[{lim_lo[j_idx]:.3f}, {lim_hi[j_idx]:.3f}])")

    # Reset — write state and set PD target = current position (so no spring force)
    robot.write_joint_position_to_sim_index(position=init_pos)
    robot.write_joint_velocity_to_sim_index(velocity=init_vel)
    # Set PD target to current position so there's no spring force
    robot.set_joint_position_target_index(target=init_pos)
    robot.set_joint_effort_target_index(target=torch.zeros(1, num_joints, device="cuda:0"))
    scene.reset()

    # Settle — maintain PD target = current position
    max_settle = 100
    for settle_step in range(max_settle):
        cur_pos = wp.to_torch(robot.data.joint_pos).clone()
        robot.set_joint_position_target_index(target=cur_pos)
        robot.set_joint_effort_target_index(target=torch.zeros(1, num_joints, device="cuda:0"))
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        vel_arr = to_np(robot.data.joint_vel).flatten()
        vel_rms = float(np.sqrt((vel_arr ** 2).mean()))
        vel_max = float(np.abs(vel_arr).max())
        if vel_max < 0.01 and vel_rms < 0.001:
            break
    
    settled_pos = to_np(robot.data.joint_pos).flatten().copy()
    if settle_step > 10 or vel_rms > 0.01:
        print(f"    ⚠️  settled after {settle_step+1} steps, vel_rms={vel_rms:.4f}, vel_max={vel_max:.4f}")
        print(f"        pos drift from init: {np.abs(settled_pos - to_np(init_pos).flatten()).max():.4f} rad")

    trajectory = []
    base_pos = settled_pos.copy()  # Use settled position as baseline
    
    for step in range(args.steps):
        pos_before = to_np(robot.data.joint_pos).flatten().copy()

        # Apply position target pulse — offset from baseline by a fixed amount
        target = torch.tensor(base_pos, device="cuda:0").unsqueeze(0)
        is_pulse = args.pulse_start <= step < args.pulse_end
        if is_pulse:
            # Offset this joint's target by a fixed amount (0.1 rad)
            target[0, j_idx] += 0.1
        
        robot.set_joint_position_target_index(target=target)
        robot.set_joint_effort_target_index(target=torch.zeros(1, num_joints, device="cuda:0"))
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        pos_after = to_np(robot.data.joint_pos).flatten().copy()
        vel_after = to_np(robot.data.joint_vel).flatten().copy()

        trajectory.append({
            "step": step,
            "pulse": is_pulse,
            "pos": float(pos_after[j_idx]),
            "vel": float(vel_after[j_idx]),
            "pos_delta": float(pos_after[j_idx] - pos_before[j_idx]),
            "implied_vel": float((pos_after[j_idx] - pos_before[j_idx]) / sim_dt),
            "all_pos": pos_after.tolist(),
            "all_vel": vel_after.tolist(),
        })

        if step == args.pulse_start:
            print(f"    pulse ON:  pos={pos_after[j_idx]:.6f}, vel={vel_after[j_idx]:.6f}, "
                  f"impl_vel={trajectory[-1]['implied_vel']:.4f}")
        elif step == args.pulse_end - 1:
            print(f"    pulse OFF: pos={pos_after[j_idx]:.6f}, disp={pos_after[j_idx]-j_init:.6f}")
        elif step == args.steps - 1:
            print(f"    settled:   pos={pos_after[j_idx]:.6f}, vel={vel_after[j_idx]:.6f}")

    results["tests"][j_name] = {
        "joint_index": j_idx,
        "initial_pos": j_init,
        "limits": [float(lim_lo[j_idx]), float(lim_hi[j_idx])],
        "params": params,
        "trajectory": trajectory,
    }

# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs(args.output, exist_ok=True)
suffix = f"_iters{args.solver_iters}" if args.solver_iters else ""
outfile = os.path.join(args.output, f"{args.robot}_{args.backend}{suffix}.json")
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {outfile}")

simulation_app.close()
