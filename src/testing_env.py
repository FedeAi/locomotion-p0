import sys
import os
import numpy as np

from GO1_env import Go1MujocoEnv


env = Go1MujocoEnv(render_mode='human')

obs = env.reset()

for _ in range(500):

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated:
        obs = env.reset()

env.close()
