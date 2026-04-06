"""Render trajectory videos with properly positioned camera.

Sets joint positions from recorded data and captures frames.
Uses a camera placed via USD xform to ensure correct positioning.
"""
import argparse, os, sys, json

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
os.environ.setdefault("DISPLAY", ":99")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

parser = argparse.ArgumentParser()
parser.add_argument("--data_files", nargs="+", required=True)
parser.add_argument("--labels", nargs="+", required=True)
parser.add_argument("--output", default="/tmp/sim2sim_combined.mp4")
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--fps", type=int, default=15)
args = parser.parse_args()

os.chdir('/home/horde/claw/git/IsaacLab')

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = launcher.app

import torch, numpy as np, math
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import TiledCameraCfg, TiledCamera
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

# Load all trajectories
all_trajs = []
for fpath in args.data_files:
    with open(fpath) as f:
        data = json.load(f)
    positions = np.array([s['joint_pos'] for s in data['steps']])
    all_trajs.append(positions)
    print(f"Loaded {fpath}: {positions.shape}")

n_steps = min(t.shape[0] for t in all_trajs)
n_robots = len(all_trajs)

# Setup sim
sim_cfg = SimulationCfg(
    device="cuda:0", dt=1.0/120.0,
    physics=PhysxCfg(solver_type=1, max_position_iteration_count=4, max_velocity_iteration_count=1),
)
sim = SimulationContext(sim_cfg)

# Ground + light
sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
sim_utils.DistantLightCfg(intensity=500.0, color=(1.0, 1.0, 0.95)).func(
    "/World/SunLight", sim_utils.DistantLightCfg(intensity=500.0, color=(1.0, 1.0, 0.95)))

# Create one robot per trajectory, spaced apart
spacing = 2.0
robots = []
cameras = []

for i in range(n_robots):
    x_off = (i - (n_robots - 1) / 2) * spacing

    # Robot
    robot_cfg = ArticulationCfg(
        prim_path=f"/World/env_{i}/Robot",
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
            pos=(x_off, 0.0, 0.0),
            joint_pos=KNOWN_INIT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={".*": 5000.0},
                damping={".*": 500.0},
            ),
        },
    )
    robots.append(Articulation(robot_cfg))

    # Table
    table_cfg = sim_utils.CuboidCfg(
        size=(0.8, 1.5, 0.04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.2)),
    )
    table_cfg.func(f"/World/env_{i}/Table", table_cfg, translation=(x_off - 0.55, 0.0, 0.235))

    # Per-robot camera from the front, looking at the hand area
    # Robot base is at (x_off, 0, 0), hand is roughly at (x_off - 0.5, 0, 0.6)
    cam_cfg = TiledCameraCfg(
        prim_path=f"/World/env_{i}/Camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(x_off - 0.4, -1.0, 0.6),
            rot=(1.0, 0.0, 0.0, 0.0),  # will set via USD
            convention="world",
        ),
        width=args.width,
        height=args.height,
        data_types=["rgb"],
    )
    cameras.append(TiledCamera(cam_cfg))

sim.reset()
for r in robots:
    r.reset()
for c in cameras:
    c.reset()

# Fix camera orientations via USD
from pxr import UsdGeom, Gf
stage = sim.stage
for i in range(n_robots):
    x_off = (i - (n_robots - 1) / 2) * spacing
    cam_prim = stage.GetPrimAtPath(f"/World/env_{i}/Camera")
    if cam_prim.IsValid():
        xformable = UsdGeom.Xformable(cam_prim)
        xformable.ClearXformOpOrder()
        # Camera looking from front-below towards the hand
        # Position: in front of robot, slightly below hand height
        xformable.AddTranslateOp().Set(Gf.Vec3d(x_off - 0.4, -1.0, 0.6))
        # Rotation: look towards +Y (towards robot), tilted up slightly
        # For USD camera: -Z is forward direction
        # We want to look from (x, -1, 0.6) towards (x, 0, 0.5)
        # That's roughly +Y direction, slightly down
        pitch = 5  # degrees tilt down
        xformable.AddRotateXYZOp().Set(Gf.Vec3d(90 - pitch, 0, 0))

# Step a few times to let camera settle
for _ in range(5):
    sim.step(render=True)
    for r in robots:
        r.update(sim.cfg.dt)
    for c in cameras:
        c.update(sim.cfg.dt)

# Get joint orderings
reorders = []
for r in robots:
    jnames = list(r.joint_names)
    if jnames == PHYSX_JOINT_ORDER:
        reorders.append(None)
    else:
        reorders.append([jnames.index(p) for p in PHYSX_JOINT_ORDER])

print(f"Rendering {n_steps} steps, {n_robots} robots...")

import cv2

all_frames = []
for step in range(n_steps):
    for i, (robot, traj) in enumerate(zip(robots, all_trajs)):
        pos_physx = traj[step]
        reorder = reorders[i]
        jnames = list(robot.joint_names)
        if reorder is not None:
            pos_native = np.zeros(len(jnames))
            for physx_i, native_i in enumerate(reorder):
                pos_native[native_i] = pos_physx[physx_i]
        else:
            pos_native = pos_physx

        pos_tensor = torch.tensor([pos_native], dtype=torch.float32, device="cuda:0")
        robot.set_joint_position_target(pos_tensor)
        robot.write_data_to_sim()

    sim.step(render=True)
    for r in robots:
        r.update(sim.cfg.dt)
    for c in cameras:
        c.update(sim.cfg.dt)

    # Capture per-robot frames and stitch
    panels = []
    for i, cam in enumerate(cameras):
        rgb = cam.data.output["rgb"]
        if isinstance(rgb, torch.Tensor):
            frame = rgb[0].cpu().numpy()
        else:
            import warp as wp
            frame = wp.to_torch(rgb)[0].cpu().numpy()
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        frame = frame.astype(np.uint8)

        # Add label
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_bgr, args.labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_bgr, args.labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(frame_bgr, f"Step {step}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        panels.append(frame_rgb)

    combined = np.hstack(panels)
    all_frames.append(combined)

    if step % 20 == 0:
        # Check first frame
        if step == 0:
            print(f"  Frame shape: {combined.shape}, mean: {combined.mean():.1f}")
        print(f"  Step {step}/{n_steps}")

# Write video
print(f"Writing {len(all_frames)} frames to {args.output}")
writer = imageio.get_writer(args.output, fps=args.fps)
for f in all_frames:
    writer.append_data(f)
writer.close()
print(f"Done: {os.path.getsize(args.output)/1024:.0f}KB")

# Also save a frame for preview
cv2.imwrite('/tmp/sim2sim_preview.png', cv2.cvtColor(all_frames[50], cv2.COLOR_RGB2BGR))
print("Preview saved to /tmp/sim2sim_preview.png")

sim.stop()
sim_app.close()
