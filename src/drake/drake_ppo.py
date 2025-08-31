import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime
from cs175.quadrotor_env import QuadrotorEnv
import argparse

import argparse

parser = argparse.ArgumentParser(description="Example script with --load flag")
parser.add_argument('--load', nargs='?', const='default_value', help="Load with an optional string")

args = parser.parse_args()

load_val: str | None = args.load

# Example placeholder - setup for future waypoints env
# env = FlattenWaypointEnv(gymnasium.make("PyFlyt/QuadX-Waypoints-v3"), context_length=1)

# Quadrotor LQR example notebook:
# https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/quadrotor-ac8fa093ca5a4894a618ada316f2750b

writer = SummaryWriter()

experiment_name='drake_ppo_hover_ascending'
experiment_logdir = f"runs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def make_env():
    # env = gymnasium.make("PyFlyt/QuadX-Hover-v4", render_mode='rgb_array', flight_mode=-1, agent_hz=100)
    env = QuadrotorEnv.build(None, np.array([0,0,1]), np.zeros(3))
    return env
env = make_vec_env(make_env, n_envs=1)
# Define a trigger function (e.g., record a video every 20,000 steps)

# video_trigger = lambda step: step % 2e4 == 0 or step == 10
# env = TensorboardVideoRecorder(env=env,
#                                 video_trigger=video_trigger,
#                                 video_length=500,
#                                 fps=30,
#                                 tb_log_dir=experiment_logdir)

if load_val is not None:
    print(f"loading {load_val}.zip")
    model = PPO.load(load_val, env, tensorboard_log=experiment_logdir, device='cpu')
else:
    model = PPO("MlpPolicy", env, tensorboard_log=experiment_logdir, device='cpu')
    
model.learn(total_timesteps=10_000_000)
model.save(f'{experiment_logdir}/weights')
