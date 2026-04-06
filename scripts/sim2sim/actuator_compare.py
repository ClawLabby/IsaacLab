"""Compare ImplicitActuator vs IdealPDActuator on PhysX.

Same Kp/Kd, same targets, same initial state.
Measures whether switching actuator model changes PhysX dynamics.
"""
import os, json
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--actuator", choices=["implicit", "ideal_pd"], required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--data_file", required=True, help="JSON with joint targets to replay")
args = parser.parse_args()

os.chdir('/home/horde/claw/git/IsaacLab')

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
sim_app = launcher.app

import torch, numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab_physx.physics import PhysxCfg

PHYSX_JOINT_ORDER = [
    "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3",
    "iiwa7_joint_4", "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
    "index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
    "index_joint_1", "middle_joint_1", "ring_joint_1", "thumb_joint_1",
    "index_joint_2", "middle_joint_2", "ring_joint_2", "thumb_joint_2",
    "index_joint_3", "middle_joint_3", "ring_joint_3", "thumb_joint_3",
]

KNOWN_INIT_POS = {
    "iiwa7_joint_1": 0.0, "iiwa7_joint_2": 0.0, "iiwa7_joint_3": 0.7854,
    "iiwa7_joint_4": 1.5708, "iiwa7_joint_5": -1.5708,
    "iiwa7_joint_6": -1.5708, "iiwa7_joint_7": 0.0,
    "index_joint_0": 0.0, "index_joint_1": 0.3, "index_joint_2": 0.3, "index_joint_3": 0.3,
    "middle_joint_0": 0.0, "middle_joint_1": 0.3, "middle_joint_2": 0.3, "middle_joint_3": 0.3,
    "ring_joint_0": 0.0, "ring_joint_1": 0.3, "ring_joint_2": 0.3, "ring_joint_3": 0.3,
    "thumb_joint_0": 1.5, "thumb_joint_1": 0.60147215, "thumb_joint_2": 0.33795027, "thumb_joint_3": 0.60845138,
}

GAINS = {
    "stiffness": {
        "iiwa7_joint_(1|2|3|4)": 300.0,
        "iiwa7_joint_5": 100.0, "iiwa7_joint_6": 50.0, "iiwa7_joint_7": 25.0,
        "(index|middle|ring|thumb)_joint_(0|1|2|3)": 3.0,
    },
    "damping": {
        "iiwa7_joint_(1|2|3|4)": 45.0,
        "iiwa7_joint_5": 20.0, "iiwa7_joint_6": 15.0, "iiwa7_joint_7": 15.0,
        "(index|middle|ring|thumb)_joint_(0|1|2|3)": 0.1,
    },
    "effort_limit_sim": {
        "iiwa7_joint_(1|2|3|4|5|6|7)": 300.0,
        "(index|middle|ring|thumb)_joint_(0|1|2|3)": 0.5,
    },
}

# Choose actuator
if args.actuator == "implicit":
    act_cfg = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        **GAINS,
        friction={"iiwa7_joint_(1|2|3|4|5|6|7)": 1.0, "(index|middle|ring|thumb)_joint_(0|1|2|3)": 0.01},
        armature={".*": 0.01},
    )
else:
    act_cfg = IdealPDActuatorCfg(
        joint_names_expr=[".*"],
        stiffness=GAINS["stiffness"],
        damping=GAINS["damping"],
        effort_limit=GAINS["effort_limit_sim"],
    )

print(f"Using {args.actuator} actuator on PhysX")

# Load targets
with open(args.data_file) as f:
    data = json.load(f)
targets = data.get('joint_targets', [s['joint_target'] for s in data.get('steps', [])])
print(f"Loaded {len(targets)} target steps")

# Setup sim
sim_cfg = SimulationCfg(
    device="cuda:0", dt=1.0/120.0,
    physics=PhysxCfg(solver_type=1, max_position_iteration_count=32, max_velocity_iteration_count=1),
)
sim = SimulationContext(sim_cfg)

robot_cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/KukaAllegro/kuka.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=32, solver_velocity_iteration_count=1,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), joint_pos=KNOWN_INIT_POS, joint_vel={".*": 0.0}),
    actuators={"all": act_cfg},
)
robot = Articulation(robot_cfg)

# Table
table_cfg = sim_utils.CuboidCfg(
    size=(0.8, 1.5, 0.04),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    collision_props=sim_utils.CollisionPropertiesCfg(),
)
table_cfg.func("/World/Table", table_cfg, translation=(-0.55, 0.0, 0.235))
sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

sim.reset()
robot.reset()

jnames = list(robot.joint_names)
print(f"Joint names: {jnames}")

# Force exact initial state
import warp as wp
init_pos_list = [KNOWN_INIT_POS.get(jn, 0.0) for jn in jnames]
init_pos_t = torch.tensor([init_pos_list], dtype=torch.float32, device="cuda:0")
init_vel_t = torch.zeros_like(init_pos_t)
robot.write_joint_state_to_sim(init_pos_t, init_vel_t)
robot.set_joint_position_target(init_pos_t)
robot.write_data_to_sim()
sim.step(render=False)
robot.update(sim.cfg.dt)

# Verify initial state
pos0 = wp.to_torch(robot.data.joint_pos)[0].cpu().numpy()
print(f"Initial arm pos: {[f'{x:.4f}' for x in pos0[:7]]}")

results = {"actuator": args.actuator, "joint_names_physx_order": PHYSX_JOINT_ORDER, "steps": []}

for step_i in range(len(targets)):
    tgt = torch.tensor([targets[step_i]], dtype=torch.float32, device="cuda:0")
    robot.set_joint_position_target(tgt)
    robot.write_data_to_sim()

    # 2 physics steps (decimation=2)
    for _ in range(2):
        sim.step(render=False)
    robot.update(sim.cfg.dt * 2)

    import warp as wp
    pos = wp.to_torch(robot.data.joint_pos)[0].cpu().numpy().tolist()
    vel = wp.to_torch(robot.data.joint_vel)[0].cpu().numpy().tolist()

    results["steps"].append({
        "joint_pos": pos,
        "joint_vel": vel,
        "joint_target": targets[step_i],
    })

with open(args.output, 'w') as f:
    json.dump(results, f)
print(f"Saved {len(results['steps'])} steps to {args.output}")

sim.stop()
sim_app.close()
