# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.envs.mdp import observations as mdp_obs
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg

from ... import dexsuite_env_cfg as dexsuite_state_impl
from ... import mdp
from . import camera_cfg
from . import dexsuite_kuka_allegro_env_cfg as kuka_allegro_dexsuite


@configclass
class KukaAllegroSingleTiledCameraSceneCfg(PresetCfg):
    """Dexsuite scene for multi-objects Lifting/Reorientation"""

    @configclass
    class KukaAllegroSingleTiledCameraSceneCfg(kuka_allegro_dexsuite.KukaAllegroSceneCfg.KukaAllegroSceneCfg):
        """Dexsuite scene for multi-objects Lifting/Reorientation"""

        base_camera: camera_cfg.BaseTiledCameraCfg = camera_cfg.BaseTiledCameraCfg()

    default = KukaAllegroSingleTiledCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)


@configclass
class KukaAllegroDuoTiledCameraSceneCfg(PresetCfg):
    """Dexsuite scene for multi-objects Lifting/Reorientation"""

    @configclass
    class KukaAllegroDuoTiledCameraSceneCfg(KukaAllegroSingleTiledCameraSceneCfg.KukaAllegroSingleTiledCameraSceneCfg):
        """Dexsuite scene for multi-objects Lifting/Reorientation"""

        wrist_camera: camera_cfg.WristTiledCameraCfg = camera_cfg.WristTiledCameraCfg()

    default = KukaAllegroDuoTiledCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)


@configclass
class KukaAllegroSingleCameraObservationsCfg(camera_cfg.StateObservationCfg):
    """Observation specifications for the MDP (CNN variant - uses raw images)."""

    @configclass
    class BaseImageObsCfg(ObsGroup):
        """Camera observations for policy group (using raw images with vision_camera)."""

        object_observation_b = ObsTerm(
            func=mdp.vision_camera,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("base_camera")},
        )

    base_image: BaseImageObsCfg = BaseImageObsCfg()

    def __post_init__(self):
        super().__post_init__()
        for group in self.__dataclass_fields__.values():
            obs_group = getattr(self, group.name)
            obs_group.history_length = None


@configclass
class KukaAllegroSingleCameraResNetObservationsCfg(camera_cfg.StateObservationCfg):
    """Observation specifications for the MDP (ResNet variant - uses ResNet18 features)."""

    @configclass
    class ResNetFeaturesObsCfg(ObsGroup):
        """ResNet18 feature extraction for policy group.

        This observation group uses a frozen, pretrained ResNet18 model (ImageNet weights)
        to extract 512-dimensional features from camera images. The ResNet is framework-agnostic
        and runs at the observation level, making it compatible with any RL framework.

        The model is automatically downloaded and cached by torchvision on first use.
        """

        resnet_features = ObsTerm(
            func=mdp_obs.image_features,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "sensor_cfg": SceneEntityCfg("base_camera"),
                "data_type": "rgb",  # ResNet models are RGB-trained; depth/albedo presets won't work well
                "model_name": "resnet18",
                # model_device defaults to env.device if not specified
            },
        )

    resnet_features: ResNetFeaturesObsCfg = ResNetFeaturesObsCfg()

    def __post_init__(self):
        super().__post_init__()
        for group in self.__dataclass_fields__.values():
            obs_group = getattr(self, group.name)
            obs_group.history_length = None


@configclass
class KukaAllegroSingleCameraResNetAugObservationsCfg(camera_cfg.StateObservationCfg):
    """Observation config with ResNet18 features + DextrAH-style image augmentation.

    Image-level domain randomization (brightness, contrast, saturation jitter + noise)
    is applied before the ResNet forward pass. This works with replicate_physics=True
    and does not require Replicator or USD material modification.
    """

    @configclass
    class ResNetFeaturesAugObsCfg(ObsGroup):
        """ResNet18 feature extraction with image-level augmentation."""

        resnet_features = ObsTerm(
            func=mdp_obs.image_features,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "sensor_cfg": SceneEntityCfg("base_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
                "inference_kwargs": {
                    "image_augmentation": {
                        "brightness": (0.6, 1.4),
                        "contrast": (0.6, 1.4),
                        "saturation": (0.5, 1.5),
                        "noise_std": 0.02,
                    },
                },
            },
        )

    resnet_features: ResNetFeaturesAugObsCfg = ResNetFeaturesAugObsCfg()

    def __post_init__(self):
        super().__post_init__()
        for group in self.__dataclass_fields__.values():
            obs_group = getattr(self, group.name)
            obs_group.history_length = None


@configclass
class KukaAllegroDuoCameraObservationsCfg(KukaAllegroSingleCameraObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class WristImageObsCfg(ObsGroup):
        wrist_observation = ObsTerm(
            func=mdp.vision_camera,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )

    wrist_image: WristImageObsCfg = WristImageObsCfg()


@configclass
class KukaAllegroSingleCameraMixinCfg(kuka_allegro_dexsuite.KukaAllegroMixinCfg):
    scene = KukaAllegroSingleTiledCameraSceneCfg()
    observations: KukaAllegroSingleCameraObservationsCfg = KukaAllegroSingleCameraObservationsCfg()

    def __post_init__(self: kuka_allegro_dexsuite.DexsuiteKukaAllegroLiftEnvCfg):
        super().__post_init__()


@configclass
class KukaAllegroSingleCameraResNetMixinCfg(kuka_allegro_dexsuite.KukaAllegroMixinCfg):
    """Mixin config for ResNet18 feature-based observations (framework-agnostic, frozen ResNet)."""

    scene = KukaAllegroSingleTiledCameraSceneCfg()
    observations: KukaAllegroSingleCameraResNetObservationsCfg = KukaAllegroSingleCameraResNetObservationsCfg()

    def __post_init__(self: kuka_allegro_dexsuite.DexsuiteKukaAllegroLiftEnvCfg):
        super().__post_init__()


@configclass
class KukaAllegroSingleCameraResNetAugMixinCfg(kuka_allegro_dexsuite.KukaAllegroMixinCfg):
    """Mixin config for ResNet18 with DextrAH-style image augmentation.

    Uses image-level domain randomization (brightness, contrast, saturation jitter)
    applied in the ResNet inference pipeline. Works with replicate_physics=True.
    """

    scene = KukaAllegroSingleTiledCameraSceneCfg()
    observations: KukaAllegroSingleCameraResNetAugObservationsCfg = KukaAllegroSingleCameraResNetAugObservationsCfg()

    def __post_init__(self: kuka_allegro_dexsuite.DexsuiteKukaAllegroLiftEnvCfg):
        super().__post_init__()


@configclass
class KukaAllegroDuoCameraMixinCfg(kuka_allegro_dexsuite.KukaAllegroMixinCfg):
    scene = KukaAllegroDuoTiledCameraSceneCfg()
    observations: KukaAllegroDuoCameraObservationsCfg = KukaAllegroDuoCameraObservationsCfg()

    def __post_init__(self: kuka_allegro_dexsuite.DexsuiteKukaAllegroLiftEnvCfg):
        super().__post_init__()


# SingleCamera (CNN variant)
@configclass
class DexsuiteKukaAllegroLiftSingleCameraEnvCfg(
    KukaAllegroSingleCameraMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg
):
    pass


# SingleCamera (ResNet variant)
@configclass
class DexsuiteKukaAllegroLiftSingleCameraResNetEnvCfg(
    KukaAllegroSingleCameraResNetMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg
):
    pass


@configclass
class DexsuiteKukaAllegroLiftSingleCameraResNetEnvCfg_PLAY(
    KukaAllegroSingleCameraResNetMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg_PLAY
):
    pass


# SingleCamera (ResNet + image augmentation variant)
@configclass
class DexsuiteKukaAllegroLiftSingleCameraResNetAugEnvCfg(
    KukaAllegroSingleCameraResNetAugMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg
):
    pass


@configclass
class DexsuiteKukaAllegroLiftSingleCameraResNetAugEnvCfg_PLAY(
    KukaAllegroSingleCameraResNetMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg_PLAY
):
    """Play config uses the non-augmented ResNet (no DR at eval time)."""

    pass


# DuoCamera
@configclass
class DexsuiteKukaAllegroLiftDuoCameraEnvCfg(KukaAllegroDuoCameraMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg):
    pass
