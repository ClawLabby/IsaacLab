"""Render 3D video of robot following recorded trajectories.

Uses Isaac Sim's rendering to show the robot executing the recorded
joint positions from each backend, one at a time.
"""
import argparse, os, sys, json

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", required=True)
parser.add_argument("--label", default="")
parser.add_argument("--output", default="/tmp/robot_video.mp4")
parser.add_argument("--cam_width", type=int, default=640)
parser.add_argument("--cam_height", type=int, default=480)
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()

os.chdir('/home/horde/claw/git/IsaacLab')

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = launcher.app

import torch, numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sensors import TiledCameraCfg, TiledCamera
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
import imageio

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

# Load data
with open(args.data_file) as f:
    data = json.load(f)
positions = np.array([s['joint_pos'] for s in data['steps']])
print(f"Loaded {positions.shape[0]} steps from {args.data_file}")

# Setup sim (using PhysX just for rendering — we set joint positions directly)
sim_cfg = SimulationCfg(
    device="cuda:0", dt=1.0/120.0,
    physics=PhysxCfg(solver_type=1, max_position_iteration_count=4, max_velocity_iteration_count=1),
)
sim = SimulationContext(sim_cfg)

# Robot
robot_cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/KukaAllegro/kuka.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), joint_pos=KNOWN_INIT_POS, joint_vel={".*": 0.0}),
    actuators={
        "all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness={".*": 5000.0}, damping={".*": 500.0}),
    },
)
robot = Articulation(robot_cfg)

# Table
table_cfg = sim_utils.CuboidCfg(
    size=(0.8, 1.5, 0.04),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
)
table_cfg.func("/World/Table", table_cfg, translation=(-0.55, 0.0, 0.235))

# Camera
cam_cfg = TiledCameraCfg(
    prim_path="/World/Camera",
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        horizontal_aperture=20.955,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.2, -1.3, 0.7),
        rot=(0.92, 0.38, 0.0, 0.0),
        convention="world",
    ),
    width=args.cam_width,
    height=args.cam_height,
    data_types=["rgb"],
)
camera = TiledCamera(cam_cfg)

# Lights
sim_utils.DomeLightCfg(intensity=1500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.0))
sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

sim.reset()
robot.reset()
camera.reset()

# Get joint ordering
joint_names = list(robot.joint_names)
reorder = None
if joint_names != PHYSX_JOINT_ORDER:
    reorder = [joint_names.index(p) for p in PHYSX_JOINT_ORDER]

print(f"Joint names: {joint_names}")
print(f"Rendering {positions.shape[0]} frames...")

frames = []
for step in range(positions.shape[0]):
    pos_physx = positions[step]

    # Convert to native order
    if reorder is not None:
        pos_native = np.zeros(len(joint_names))
        for physx_i, native_i in enumerate(reorder):
            pos_native[native_i] = pos_physx[physx_i]
    else:
        pos_native = pos_physx

    # Set joint positions directly (high stiffness actuator tracks instantly)
    pos_tensor = torch.tensor([pos_native], dtype=torch.float32, device="cuda:0")
    robot.set_joint_position_target(pos_tensor)

    robot.write_data_to_sim()
    sim.step(render=True)
    robot.update(sim.cfg.dt)
    camera.update(sim.cfg.dt)

    # Capture frame
    rgb = camera.data.output["rgb"]
    if isinstance(rgb, torch.Tensor):
        frame = rgb[0].cpu().numpy()
    else:
        import warp as wp
        frame = wp.to_torch(rgb)[0].cpu().numpy()

    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    frames.append(frame.astype(np.uint8))

    if step % 20 == 0:
        print(f"  Frame {step}/{positions.shape[0]}")

# Write video
print(f"Writing {len(frames)} frames to {args.output}")
writer = imageio.get_writer(args.output, fps=args.fps)
for f in frames:
    writer.append_data(f)
writer.close()
print(f"Done! {os.path.getsize(args.output)/1024:.0f}KB")

sim.stop()
sim_app.close()
