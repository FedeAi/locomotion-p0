import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from github_env import Go1MujocoEnv


env = Go1MujocoEnv(render_mode='human')

obs = env.reset()

for _ in range(500):

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated:
        obs = env.reset()

env.close()
