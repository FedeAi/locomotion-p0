import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import gymnasium as gym
from gym_custom_envs.o2_env import AntEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

class VisualizeCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(VisualizeCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.env  = make_vec_env(lambda: AntEnv(render_mode="human"), n_envs=1)
            
            obs = self.env.reset()
            for _ in range(200):
                action, _states = model.predict(obs)
                result = self.env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                else:
                    obs, reward, done, info = result
                    terminated, truncated = done, done
                self.env.render()
                if terminated or truncated:
                    obs = self.env.reset()
            
            self.env.close()  # Close the viewer window specifically
        
        return True
    
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            self.logger.record("env/reward_forward", info.get("reward_forward", 0))
            self.logger.record("env/reward_ctrl", info.get("reward_ctrl", 0))
            self.logger.record("env/reward_contact", info.get("reward_contact", 0))
            self.logger.record("env/reward_survive", info.get("reward_survive", 0))
            self.logger.record("env/reward_action_rate", info.get("reward_action_rate", 0))
        return True
# Create the environment
env = make_vec_env(lambda: AntEnv(render_mode=None), n_envs=1)

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq=20000, save_path="./checkpoints", verbose=1):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # create directory if not exist
            os.makedirs(self.save_path, exist_ok=True)
            checkpoint_file = os.path.join(self.save_path, f"ppo_ant_{self.n_calls}_steps")
            self.model.save(checkpoint_file)
            if self.verbose > 0:
                print(f"Model checkpoint saved: {checkpoint_file}")
        return True


# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_o2_tensorboard", policy_kwargs=dict(net_arch=[128, 128, 128]))
# Reload a model if there is a saved one, make sure to set the environment correctly (it is saved as ppo_ant.zip)
if os.path.exists("ppo_ant.zip"):
    model = PPO.load("ppo_ant", env=env)

# Train the model with the visualization callback
visualize_callback = VisualizeCallback(check_freq=20000)
tensorboard_callback = TensorboardCallback()
checkpoint_callback = CheckpointCallback()

model.learn(total_timesteps=5000000, callback=[visualize_callback, tensorboard_callback, checkpoint_callback])


# Save the model
model.save("ppo_ant")

# Load the model
model = PPO.load("ppo_ant")


# Test the trained model
render_env = make_vec_env(lambda: AntEnv(render_mode="human"), n_envs=1)

obs = render_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    result = render_env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    else:
        obs, reward, done, info = result
        terminated, truncated = done, done
    render_env.render()
    if terminated or truncated:
        obs = render_env.reset()
render_env.close()