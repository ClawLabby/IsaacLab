# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp
from .camera_cfg import (
    BaseTiledCameraCfg,
    DuoCameraObservationsCfg,
    SingleCameraObservationsCfg,
    SingleCameraJitterObservationsCfg,
    StateObservationCfg,
    WristTiledCameraCfg,
)

FINGERTIP_LIST = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
THUMB_SENSOR = "thumb_link_3_object_s"
FINGER_SENSORS = [f"{name}_object_s" for name in FINGERTIP_LIST if name != "thumb_link_3"]


@configclass
class KukaAllegroPhysicsCfg(PresetCfg):
    default = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_max_rigid_patch_count=4 * 5 * 2**15,
        gpu_found_lost_pairs_capacity=2**26,
    )
    newton = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            ls_parallel=False,
            use_mujoco_contacts=False,
            ccd_iterations=5000,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    physx = default


@configclass
class KukaAllegroSceneCfg(PresetCfg):
    @configclass
    class KukaAllegroSceneCfg(dexsuite.SceneCfg):
        """Kuka Allegro participant scene for Dexsuite Lifting/Reorientation"""

        robot: ArticulationCfg = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        base_camera: TiledCameraCfg | None = None

        wrist_camera: TiledCameraCfg | None = None

        def __post_init__(self: dexsuite.SceneCfg):
            super().__post_init__()
            for link_name in FINGERTIP_LIST:
                setattr(
                    self,
                    f"{link_name}_object_s",
                    ContactSensorCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    ),
                )

    default = KukaAllegroSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)
    single_camera = default.replace(base_camera=BaseTiledCameraCfg())
    single_camera_visual_dr = default.replace(base_camera=BaseTiledCameraCfg(), replicate_physics=False)
    duo_camera = default.replace(base_camera=BaseTiledCameraCfg(), wrist_camera=WristTiledCameraCfg())


@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroReorientRewardCfg(dexsuite.RewardsCfg):
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 0.1, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=1.0,
        params={
            "threshold": 0.01,
            "sensor_names": FINGER_SENSORS + [THUMB_SENSOR],
        },
    )

    def __post_init__(self: dexsuite.RewardsCfg):
        super().__post_init__()
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])
        self.fingers_to_object.params["thumb_name"] = THUMB_SENSOR
        self.fingers_to_object.params["finger_names"] = FINGER_SENSORS
        self.position_tracking.params["thumb_name"] = THUMB_SENSOR
        self.position_tracking.params["finger_names"] = FINGER_SENSORS
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = THUMB_SENSOR
            self.orientation_tracking.params["finger_names"] = FINGER_SENSORS
        self.success.params["thumb_name"] = THUMB_SENSOR
        self.success.params["finger_names"] = FINGER_SENSORS


@configclass
class KukaAllegroObservationCfg(PresetCfg):
    state = StateObservationCfg()
    single_camera = SingleCameraObservationsCfg()
    single_camera_jitter = SingleCameraJitterObservationsCfg()
    duo_camera = DuoCameraObservationsCfg()
    default = state


@configclass
class KukaAllegroEventCfg(PresetCfg):
    @configclass
    class KukaAllegroPhysxEventCfg(dexsuite.StartupEventCfg, dexsuite.EventCfg):
        pass

    @configclass
    class NewtonVisualDREventCfg(dexsuite.EventCfg):
        """Newton events + Newton Warp renderer shape color randomization."""

        randomize_shape_colors = EventTerm(
            func=mdp.randomize_newton_shape_colors,
            mode="reset",
            params={
                "sensor_name": "base_camera",
                "color_range": (0.1, 0.9),
            },
        )

    @configclass
    class PhysxVisualDREventCfg(dexsuite.StartupEventCfg, dexsuite.EventCfg):
        """PhysX events + RTX color randomization via Replicator."""

        randomize_object_color = EventTerm(
            func=mdp.randomize_visual_color,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mesh_name": ".*",
                "event_name": "randomize_object_color",
                "colors": {"r": (0.1, 0.9), "g": (0.1, 0.9), "b": (0.1, 0.9)},
            },
        )

        randomize_table_color = EventTerm(
            func=mdp.randomize_visual_color,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("table"),
                "mesh_name": ".*",
                "event_name": "randomize_table_color",
                "colors": {"r": (0.1, 0.9), "g": (0.1, 0.9), "b": (0.1, 0.9)},
            },
        )

    default = KukaAllegroPhysxEventCfg()
    newton = dexsuite.EventCfg()
    physx = default
    physx_no_dr = dexsuite.EventCfg()
    newton_visual_dr = NewtonVisualDREventCfg()
    physx_visual_dr = PhysxVisualDREventCfg()


@configclass
class KukaAllegroMixinCfg:
    scene: KukaAllegroSceneCfg = KukaAllegroSceneCfg()
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    observations: KukaAllegroObservationCfg = KukaAllegroObservationCfg()
    events: KukaAllegroEventCfg = KukaAllegroEventCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = KukaAllegroPhysicsCfg()


@configclass
class DexsuiteKukaAllegroReorientEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroReorientEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
