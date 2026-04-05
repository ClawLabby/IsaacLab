"""Record video of cross-backend policy eval (state obs + added visualization camera)."""
import os
import sys
import numpy as np

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"

os.chdir('/home/horde/claw/git/IsaacLab')
sys.path.insert(0, '/home/horde/claw/git/IsaacLab/scripts/reinforcement_learning/rsl_rl')

import gymnasium as gym
from joint_remapper import JointRemapper

frames = []
MAX_FRAMES = 300
joint_remapper = None
remap_info = []

_orig_make = gym.make
def _patched_make(*args, **kwargs):
    global joint_remapper, remap_info
    env = _orig_make(*args, **kwargs)
    
    # Set up joint remapping
    ckpt_dir = os.path.dirname(os.environ.get('CHECKPOINT', ''))
    train_jn = JointRemapper.load_joint_names(ckpt_dir)
    eval_jn = env.unwrapped.scene['robot'].joint_names
    if train_jn is not None:
        joint_remapper = JointRemapper(train_jn, eval_jn)
        if joint_remapper.needs_remap:
            joint_remapper.print_mapping()
            nj = joint_remapper.num_joints
            obs_mgr = env.unwrapped.observation_manager
            for gn in obs_mgr.active_terms:
                tds = obs_mgr.group_obs_term_dim[gn]
                tns = obs_mgr.active_terms[gn]
                off = 0
                for tn, td in zip(tns, tds):
                    sz = 1
                    for d in td: sz *= d
                    if tn in ('joint_pos', 'joint_vel', 'actions') and sz % nj == 0:
                        remap_info.append((gn, off, sz))
                        print(f'  [REMAP] {gn}/{tn} offset={off} size={sz}', flush=True)
                    off += sz
        else:
            joint_remapper = None
    
    class VideoWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._step_count = 0
            
        def step(self, action):
            import torch
            
            # Remap actions if needed
            if joint_remapper is not None:
                action = joint_remapper.remap_actions(action)
            
            obs, rew, term, trunc, info = self.env.step(action)
            self._step_count += 1
            
            # Remap observations for next policy call
            if joint_remapper is not None:
                for gn, off, sz in remap_info:
                    chunk = obs[gn][:, off:off+sz]
                    nj = joint_remapper.num_joints
                    flat = chunk.reshape(-1, nj)
                    remapped = joint_remapper.remap_joint_obs(flat)
                    obs[gn][:, off:off+sz] = remapped.reshape(chunk.shape)
            
            if self._step_count <= MAX_FRAMES:
                unwrapped = self.unwrapped
                if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, '_sensors'):
                    for name, sensor in unwrapped.scene._sensors.items():
                        if not hasattr(sensor, 'data') or not hasattr(sensor.data, 'output'):
                            continue
                        out = sensor.data.output
                        if 'rgb' not in out:
                            continue
                        rgb = out['rgb'].cpu().numpy()
                        if len(rgb.shape) == 4:
                            n, h, w, c = rgb.shape
                            # Just take env 0 for a single clean view
                            frame = rgb[0, :, :, :3].astype(np.uint8)
                            frames.append(frame)
                            if self._step_count == 1:
                                print(f'CAPTURE: First frame shape={frame.shape}', flush=True)
                        break
                        
            if self._step_count == MAX_FRAMES:
                print(f'CAPTURE: Reached {MAX_FRAMES} frames', flush=True)
                
            return obs, rew, term, trunc, info
            
    return VideoWrapper(env)

gym.make = _patched_make

# Set up play.py args - use camera task for rendering, but load state checkpoint
# The camera task has extra perception obs that won't match the state checkpoint
# So instead, use the state task but with a camera preset added for visualization
ckpt = os.environ.get('CHECKPOINT', 'logs/rsl_rl/dexsuite_kuka_allegro/2026-03-31_21-53-28/model_9750.pt')
backend = os.environ.get('BACKEND', 'newton')

sys.argv = [
    'play.py',
    '--task', 'Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-ResNet-v0',
    '--num_envs', '1',
    '--headless',
    '--enable_cameras',
    f'presets={backend},rgb128',
]

os.environ['CHECKPOINT'] = ckpt

# We can't use the camera task with state checkpoint - obs mismatch.
# Instead patch the play.py to skip the checkpoint loading size check...
# Actually, let me try a different approach: use state task + add a camera sensor manually

# Let's just use Newton's ViewerGL for rendering
print("NOTE: This approach needs the Camera task. For state-only checkpoint, use record_state_video.py", flush=True)

import runpy
try:
    runpy.run_path('/home/horde/claw/git/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py', run_name='__main__')
except SystemExit:
    pass

print(f'RESULT: Captured {len(frames)} frames', flush=True)
if frames:
    import imageio
    output_path = f'/tmp/cross_eval_{backend}.mp4'
    imageio.mimsave(output_path, frames, fps=30)
    print(f'RESULT: Saved to {output_path}', flush=True)
