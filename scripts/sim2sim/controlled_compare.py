"""Controlled sim2sim comparison: single robot, forced initial state, recorded controls.

Phase 1: Run DexSuite with PhysX, force known reset, record joint targets + joint state
Phase 2: Create minimal single-robot env, replay joint targets on PhysX, compare
Phase 3: Same replay on Newton, compare

This isolates the physics response from everything else (env logic, rewards, objects, etc.)
"""
import argparse, os, sys, json

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_steps", type=int, default=200)
parser.add_argument("--phase", required=True, choices=[
    "record_dexsuite",    # Run policy in DexSuite PhysX, record everything
    "replay_standalone",  # Replay joint targets in standalone robot env
])
parser.add_argument("--backend", default="physx", choices=["physx", "newton"])
parser.add_argument("--data_file", default="/tmp/controlled_data.json")
parser.add_argument("--output", default=None)
parser.add_argument("--task", default="Isaac-Dexsuite-Kuka-Allegro-Lift-v0")
args, hydra_args = parser.parse_known_args()

if args.output is None:
    args.output = f"/tmp/controlled_{args.phase}_{args.backend}.json"

os.chdir('/home/horde/claw/git/IsaacLab')
sys.path.insert(0, 'scripts/reinforcement_learning/rsl_rl')

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=False)
sim_app = launcher.app

import torch, numpy as np, gymnasium as gym, importlib.metadata

PHYSX_JOINT_ORDER = [
    "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3",
    "iiwa7_joint_4", "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
    "index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
    "index_joint_1", "middle_joint_1", "ring_joint_1", "thumb_joint_1",
    "index_joint_2", "middle_joint_2", "ring_joint_2", "thumb_joint_2",
    "index_joint_3", "middle_joint_3", "ring_joint_3", "thumb_joint_3",
]

# Known initial joint positions (from DexSuite default + zeros for reset offset)
KNOWN_INIT_POS = {
    "iiwa7_joint_1": 0.0,
    "iiwa7_joint_2": 0.0,
    "iiwa7_joint_3": 0.7854,
    "iiwa7_joint_4": 1.5708,
    "iiwa7_joint_5": -1.5708,
    "iiwa7_joint_6": -1.5708,
    "iiwa7_joint_7": 0.0,
    "index_joint_0": 0.0, "index_joint_1": 0.3, "index_joint_2": 0.3, "index_joint_3": 0.3,
    "middle_joint_0": 0.0, "middle_joint_1": 0.3, "middle_joint_2": 0.3, "middle_joint_3": 0.3,
    "ring_joint_0": 0.0, "ring_joint_1": 0.3, "ring_joint_2": 0.3, "ring_joint_3": 0.3,
    "thumb_joint_0": 1.5, "thumb_joint_1": 0.60147215, "thumb_joint_2": 0.33795027, "thumb_joint_3": 0.60845138,
}


def get_joint_data(robot, to_physx_order=None):
    """Get joint pos/vel as numpy, optionally reorder to PhysX order."""
    jp = robot.data.joint_pos
    jv = robot.data.joint_vel
    if not hasattr(jp, 'cpu'):
        import warp as wp
        jp = wp.to_torch(jp)
        jv = wp.to_torch(jv)
    jp = jp[0].cpu().numpy()
    jv = jv[0].cpu().numpy()
    if to_physx_order is not None:
        jp = jp[to_physx_order]
        jv = jv[to_physx_order]
    return jp, jv


def get_joint_targets(robot, to_physx_order=None):
    """Get the actual joint position targets being sent to the actuator."""
    # After action processing, the target is stored in the articulation
    jt = robot.data.joint_pos_target
    if not hasattr(jt, 'cpu'):
        import warp as wp
        jt = wp.to_torch(jt)
    jt = jt[0].cpu().numpy()
    if to_physx_order is not None:
        jt = jt[to_physx_order]
    return jt


def build_reorder(joint_names):
    """Build reorder index from native joint order to PhysX order."""
    if list(joint_names) == PHYSX_JOINT_ORDER:
        return None
    return [list(joint_names).index(p) for p in PHYSX_JOINT_ORDER]


def phase_record_dexsuite():
    """Run DexSuite with PhysX, force initial state, record joint targets."""
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import isaaclab_tasks, cli_args
    installed_version = importlib.metadata.version('rsl-rl-lib')

    @hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = "cuda:0"
        agent_cfg.device = "cuda:0"
        env_cfg.seed = 42

        # Disable random joint reset offset to get deterministic init
        # Remove the random offset events
        if hasattr(env_cfg, 'events'):
            if hasattr(env_cfg.events, 'reset_robot_joints'):
                env_cfg.events.reset_robot_joints.params['position_range'] = [0.0, 0.0]
                print("Disabled random joint reset offset")
            if hasattr(env_cfg.events, 'reset_robot_wrist_joint'):
                env_cfg.events.reset_robot_wrist_joint.params['position_range'] = [0.0, 0.0]
                print("Disabled random wrist reset offset")

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        robot = env.unwrapped.scene["robot"]
        joint_names = list(robot.joint_names)
        reorder = build_reorder(joint_names)
        print(f"Joint names: {joint_names}")
        print(f"Reorder to PhysX: {reorder}")

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device="cuda:0")

        obs = env.get_observations()

        data = {
            "joint_names_physx_order": PHYSX_JOINT_ORDER,
            "joint_names_native": joint_names,
            "initial_pos": None,
            "steps": [],
        }

        for step in range(args.num_steps):
            with torch.inference_mode():
                jp, jv = get_joint_data(robot, reorder)

                if step == 0:
                    data["initial_pos"] = jp.tolist()
                    print(f"Initial joint pos (PhysX order): {np.round(jp[:7], 4)}")

                actions = policy(obs)
                raw_action = actions[0].cpu().numpy()

                obs, rew, dones, infos = env.step(actions)
                policy.reset(dones)

                # Record the joint target AFTER action processing
                jt = get_joint_targets(robot, reorder)

                # Record everything in PhysX joint order
                data["steps"].append({
                    "joint_pos": jp.tolist(),
                    "joint_vel": jv.tolist(),
                    "joint_target": jt.tolist(),
                    "raw_action": raw_action.tolist() if reorder is None else raw_action.tolist(),
                    "reward": rew[0].item(),
                })

                if step % 20 == 0:
                    print(f"  Step {step}: j1={jp[0]:+.4f}, j4={jp[3]:+.4f}, j7={jp[6]:+.4f}, rew={rew[0].item():.4f}")

                if dones[0]:
                    print(f"  Episode ended at step {step}")
                    break

        with open(args.output, 'w') as f:
            json.dump(data, f)
        print(f"\nSaved {len(data['steps'])} steps to {args.output}")

        # Also save just the joint targets for replay
        targets = [s["joint_target"] for s in data["steps"]]
        initial = data["initial_pos"]
        replay_data = {
            "joint_targets": targets,
            "initial_pos": initial,
            "joint_names_physx_order": PHYSX_JOINT_ORDER,
        }
        with open(args.data_file, 'w') as f:
            json.dump(replay_data, f)
        print(f"Saved replay data to {args.data_file}")

        env.close()
        sim_app.close()

    sys.argv = [sys.argv[0]] + hydra_args
    _run()


def phase_replay_standalone():
    """Replay joint targets on a standalone robot (no objects, no env logic)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.assets import ArticulationCfg, Articulation
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    from isaaclab.actuators import ImplicitActuatorCfg

    # Load recorded targets
    with open(args.data_file) as f:
        replay = json.load(f)
    targets = replay["joint_targets"]
    initial_pos = replay["initial_pos"]
    print(f"Loaded {len(targets)} targets, initial pos: {np.round(initial_pos[:7], 4)}")

    # Choose physics backend
    if args.backend == "newton":
        from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
        physics_cfg = NewtonCfg(
            num_substeps=2,
            solver_cfg=MJWarpSolverCfg(
                iterations=100, ls_iterations=15,
                solver="newton", integrator="implicitfast",
                nconmax=200, njmax=300,
            )
        )
    else:
        from isaaclab_physx.physics import PhysxCfg
        physics_cfg = PhysxCfg(
            solver_type=1,
            max_position_iteration_count=32,
            max_velocity_iteration_count=1,
        )

    sim_cfg = SimulationCfg(
        device="cuda:0",
        dt=1.0/120.0,  # DexSuite uses dt=1/120
        physics=physics_cfg,
    )

    sim = SimulationContext(sim_cfg)

    # Same robot config as DexSuite
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/KukaAllegro/kuka.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={k: v for k, v in KNOWN_INIT_POS.items()},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "kuka_allegro": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim={
                    "iiwa7_joint_(1|2|3|4|5|6|7)": 300.0,
                    "index_joint_(0|1|2|3)": 0.5,
                    "middle_joint_(0|1|2|3)": 0.5,
                    "ring_joint_(0|1|2|3)": 0.5,
                    "thumb_joint_(0|1|2|3)": 0.5,
                },
                stiffness={
                    "iiwa7_joint_(1|2|3|4)": 300.0,
                    "iiwa7_joint_5": 100.0, "iiwa7_joint_6": 50.0, "iiwa7_joint_7": 25.0,
                    "(index|middle|ring|thumb)_joint_(0|1|2|3)": 3.0,
                },
                damping={
                    "iiwa7_joint_(1|2|3|4)": 45.0,
                    "iiwa7_joint_5": 20.0, "iiwa7_joint_6": 15.0, "iiwa7_joint_7": 15.0,
                    "(index|middle|ring|thumb)_joint_(0|1|2|3)": 0.1,
                },
                friction={
                    "iiwa7_joint_(1|2|3|4|5|6|7)": 1.0,
                    "(index|middle|ring|thumb)_joint_(0|1|2|3)": 0.01,
                },
                armature={".*": 0.01},
            ),
        },
    )

    # Spawn ground plane
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

    # Spawn table (same as DexSuite: kinematic box at (-0.55, 0, 0.235))
    table_cfg = sim_utils.CuboidCfg(
        size=(0.8, 1.5, 0.04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visible=False,
    )
    table_cfg.func("/World/Table", table_cfg, translation=(-0.55, 0.0, 0.235))

    robot = Articulation(robot_cfg)

    sim.reset()
    robot.reset()

    # Get joint ordering
    joint_names = list(robot.joint_names)
    print(f"Standalone joint names: {joint_names}")
    reorder_to_physx = build_reorder(joint_names)

    # Build PhysX→native remap for setting targets
    if reorder_to_physx is not None:
        # physx_to_native[i] = native index for physx joint i
        physx_to_native = [0] * len(PHYSX_JOINT_ORDER)
        for pi, ni in enumerate(reorder_to_physx):
            physx_to_native[ni] = pi  # No, this is wrong
        # Actually: reorder_to_physx[i] = native_idx for PHYSX_JOINT_ORDER[i]
        # So to go from physx order to native: native[reorder_to_physx[i]] = physx[i]
        # Or equivalently: for setting targets in native order from physx-ordered data,
        # we need native_target[native_idx] = physx_target[physx_idx]
        # where native_idx = reorder_to_physx[physx_idx]... no.
        # reorder_to_physx maps: for each physx joint, what's its native index
        # So reorder_to_physx[physx_i] = native_i
        # To set native-ordered targets from physx-ordered data:
        # native_targets[reorder_to_physx[i]] = physx_targets[i] for all i
        pass

    # Force initial state
    init_pos_physx = np.array(initial_pos)
    if reorder_to_physx is not None:
        # Convert PhysX-ordered initial pos to native order
        init_pos_native = np.zeros(len(joint_names))
        for physx_i, native_i in enumerate(reorder_to_physx):
            init_pos_native[native_i] = init_pos_physx[physx_i]
        init_tensor = torch.tensor([init_pos_native], dtype=torch.float32, device="cuda:0")
    else:
        init_tensor = torch.tensor([init_pos_physx], dtype=torch.float32, device="cuda:0")

    robot.write_joint_state_to_sim(init_tensor, torch.zeros_like(init_tensor))

    # Verify initial state
    jp, jv = get_joint_data(robot, reorder_to_physx)
    print(f"Forced initial pos (PhysX order): {np.round(jp[:7], 4)}")
    print(f"Expected:                          {np.round(init_pos_physx[:7], 4)}")
    print(f"Diff: {np.abs(jp - init_pos_physx).max():.6f}")

    data = {"backend": args.backend, "steps": []}

    # DexSuite uses decimation=2, so env.step does 2 physics steps
    # We need to do the same: set target, step twice
    for step_i, target_physx in enumerate(targets):
        jp, jv = get_joint_data(robot, reorder_to_physx)

        data["steps"].append({
            "joint_pos": jp.tolist(),
            "joint_vel": jv.tolist(),
            "joint_target": target_physx,
        })

        # Set joint targets in native order
        target_arr = np.array(target_physx)
        if reorder_to_physx is not None:
            target_native = np.zeros(len(joint_names))
            for physx_i, native_i in enumerate(reorder_to_physx):
                target_native[native_i] = target_arr[physx_i]
            target_tensor = torch.tensor([target_native], dtype=torch.float32, device="cuda:0")
        else:
            target_tensor = torch.tensor([target_arr], dtype=torch.float32, device="cuda:0")

        robot.set_joint_position_target(target_tensor)

        # Step physics twice (decimation=2)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.cfg.dt)
        sim.step(render=False)
        robot.update(sim.cfg.dt)

        if step_i % 20 == 0:
            print(f"  Step {step_i}: j1={jp[0]:+.4f}, j4={jp[3]:+.4f}, j7={jp[6]:+.4f}")

    with open(args.output, 'w') as f:
        json.dump(data, f)
    print(f"\nSaved {len(data['steps'])} steps ({args.backend}) to {args.output}")

    sim.stop()
    sim_app.close()


if args.phase == "record_dexsuite":
    phase_record_dexsuite()
else:
    phase_replay_standalone()
