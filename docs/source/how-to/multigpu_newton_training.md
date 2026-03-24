# Multi-GPU Training with Newton Physics

## Overview

This fix enables multi-GPU distributed training with Newton physics and camera
rendering in Isaac Lab. Three interrelated bugs were fixed:

1. **Device assignment**: Non-rank-0 GPUs were assigned to wrong CUDA devices
2. **Camera renderer**: Newton warp renderer preset wasn't applied correctly
3. **Fabric mode**: Non-cuda:0 GPUs need fabric disabled to avoid race conditions

## Usage

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=4 \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
  --num_envs 1024 \
  --headless --distributed \
  presets=newton,cube
```

## Environment Variables

- `NCCL_P2P_DISABLE=1`: Required for containers with IOMMU (P2P transport fails)
- `NCCL_IB_DISABLE=1`: Required when InfiniBand is not available

## Known Issues

- CCD solver warnings (`opt.ccd_iterations needs to be increased`) are expected
  under high contact scenarios. The NaN watchdog handles resulting physics failures.
- `abnormal_robot` termination rate is higher with camera observations (~50%)
  vs state-only (~22%). This is a physics stability issue, not a training bug.

## Testing

```bash
# Run multi-GPU Newton physics tests
python -m pytest source/isaaclab/test/test_multigpu_newton.py -v
```
