"""Render joint trajectory comparison as video.

Loads recorded joint positions from the controlled comparison and renders
the robot in each configuration side by side (or sequentially).
"""
import argparse, os, sys, json

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
os.environ.setdefault("DISPLAY", ":99")

parser = argparse.ArgumentParser()
parser.add_argument("--data_files", nargs="+", required=True)
parser.add_argument("--labels", nargs="+", default=None)
parser.add_argument("--output", default="/tmp/sim2sim_comparison.mp4")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--backend", default="physx", choices=["physx", "newton"])
args = parser.parse_args()

if args.labels is None:
    args.labels = [os.path.basename(f).replace('.json','') for f in args.data_files]

os.chdir('/home/horde/claw/git/IsaacLab')

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = launcher.app

import torch, numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import CameraCfg, Camera
import imageio

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

# Load all trajectory data
all_trajs = []
for fpath in args.data_files:
    with open(fpath) as f:
        data = json.load(f)
    positions = np.array([s['joint_pos'] for s in data['steps']])
    all_trajs.append(positions)
    print(f"Loaded {fpath}: {positions.shape[0]} steps")

n_steps = min(t.shape[0] for t in all_trajs)
n_robots = len(all_trajs)
print(f"Comparing {n_robots} trajectories, {n_steps} steps")

# Setup sim
if args.backend == "newton":
    from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
    physics_cfg = NewtonCfg(num_substeps=2, solver_cfg=MJWarpSolverCfg(iterations=100, ls_iterations=15))
else:
    from isaaclab_physx.physics import PhysxCfg
    physics_cfg = PhysxCfg(solver_type=1, max_position_iteration_count=4, max_velocity_iteration_count=1)

sim_cfg = SimulationCfg(device="cuda:0", dt=1.0/120.0, physics=physics_cfg)
sim = SimulationContext(sim_cfg)

# Spawn robots side by side
spacing = 1.5
robots = []
for i in range(n_robots):
    x_offset = (i - (n_robots-1)/2) * spacing
    robot_cfg = ArticulationCfg(
        prim_path=f"/World/Robot_{i}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/KukaAllegro/kuka.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, retain_accelerations=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(x_offset, 0.0, 0.0),
            joint_pos=KNOWN_INIT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={".*": 1000.0},  # High stiffness to track targets precisely
                damping={".*": 100.0},
            ),
        },
    )
    robots.append(Articulation(robot_cfg))

# Spawn tables
for i in range(n_robots):
    x_offset = (i - (n_robots-1)/2) * spacing
    table_cfg = sim_utils.CuboidCfg(
        size=(0.8, 1.5, 0.04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visible=True,
    )
    table_cfg.func(f"/World/Table_{i}", table_cfg, translation=(x_offset - 0.55, 0.0, 0.235))

# Lights
sim_utils.DomeLightCfg(intensity=1500.0).func("/World/DomeLight", sim_utils.DomeLightCfg(intensity=1500.0))

# Ground
sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

sim.reset()
for r in robots:
    r.reset()

# Get joint name mapping for each robot
reorders = []
for r in robots:
    jnames = list(r.joint_names)
    if jnames == PHYSX_JOINT_ORDER:
        reorders.append(None)
    else:
        reorders.append([jnames.index(p) for p in PHYSX_JOINT_ORDER])

# Setup viewport camera via USD
from pxr import UsdGeom, Gf
stage = sim.stage
cam_path = "/World/ViewCamera"
cam_prim = UsdGeom.Camera.Define(stage, cam_path)
# Position camera to see all robots
cam_x = 0.0
cam_y = -1.8
cam_z = 1.0
cam_prim.GetFocalLengthAttr().Set(18.0)

import omni.kit.viewport.utility as viewport_utils
viewport = viewport_utils.get_active_viewport()
viewport.set_active_camera(cam_path)

# Set camera transform
xform = UsdGeom.Xformable(cam_prim)
xform.ClearXformOpOrder()
xform.AddTranslateOp().Set(Gf.Vec3d(cam_x, cam_y, cam_z))
# Look at origin
import math
pitch = math.degrees(math.atan2(cam_z - 0.4, abs(cam_y)))
xform.AddRotateXYZOp().Set(Gf.Vec3d(90 - pitch, 0, 0))

print("Rendering video...")
frames = []

for step in range(n_steps):
    # Set joint positions from recorded data
    for i, (robot, traj) in enumerate(zip(robots, all_trajs)):
        jnames = list(robot.joint_names)
        pos_physx = traj[step]

        # Convert from PhysX order to native order
        reorder = reorders[i]
        if reorder is not None:
            pos_native = np.zeros(len(jnames))
            for physx_i, native_i in enumerate(reorder):
                pos_native[native_i] = pos_physx[physx_i]
        else:
            pos_native = pos_physx

        pos_tensor = torch.tensor([pos_native], dtype=torch.float32, device="cuda:0")
        vel_tensor = torch.zeros_like(pos_tensor)
        robot.write_joint_state_to_sim(pos_tensor, vel_tensor)

    # Step sim for rendering
    robot.write_data_to_sim()
    sim.step(render=True)
    for r in robots:
        r.update(sim.cfg.dt)

    # Capture frame from viewport
    try:
        from omni.kit.viewport.utility import capture_viewport_to_buffer
        buffer = capture_viewport_to_buffer(viewport)
        if buffer is not None:
            frame = np.frombuffer(buffer, dtype=np.uint8).reshape(viewport.resolution[1], viewport.resolution[0], 4)
            frames.append(frame[:, :, :3])  # Drop alpha
    except Exception as e:
        if step == 0:
            print(f"Viewport capture failed: {e}")
            print("Falling back to no-video mode")
            break

    if step % 20 == 0:
        print(f"  Rendered step {step}/{n_steps}")

if frames:
    print(f"Writing {len(frames)} frames to {args.output}")
    writer = imageio.get_writer(args.output, fps=args.fps)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"Done! {os.path.getsize(args.output)/1024:.0f}KB")
else:
    print("No frames captured - trying screenshot approach")

sim.stop()
sim_app.close()
