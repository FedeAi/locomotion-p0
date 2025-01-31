import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from github_env import Go1MujocoEnv
from tqdm import tqdm
import numpy as np

MODEL_DIR = "models"
LOG_DIR = "logs"



from stable_baselines3.common.callbacks import BaseCallback




class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            self.logger.record("env/linear_vel_tracking_reward", info.get("linear_vel_tracking_reward", 0))
            self.logger.record("env/angular_vel_tracking_reward", info.get("angular_vel_tracking_reward", 0))
            self.logger.record("env/feet_air_time_reward", info.get("feet_air_time_reward", 0))
            self.logger.record("env/reward_ctrl", info.get("reward_ctrl", 0))
            self.logger.record("env/action_rate_cost", info.get("action_rate_cost", 0))
            self.logger.record("env/vertical_vel_cost", info.get("vertical_vel_cost", 0)) 
            self.logger.record("env/xy_angular_vel_cost", info.get("xy_angular_vel_cost", 0)) 
            self.logger.record("env/joint_limit_cost", info.get("joint_limit_cost", 0))
            self.logger.record("env/joint_acceleration_cost", info.get("joint_acceleration_cost", 0))
            self.logger.record("env/orientation_cost", info.get("orientation_cost", 0)) 
            self.logger.record("env/default_joint_position_cost", info.get("default_joint_position_cost", 0)) 
        return True    
    

class VisualizeCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(VisualizeCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:

        # Check if the current step is a multiple of the frequency
        if self.n_calls % self.check_freq == 0:

            print(f"[VisualizeCallback] Rendering at step {self.n_calls}")

            # Create a single environment with rendering enabled
            env = Go1MujocoEnv(render_mode="human")
            
            # Reset the environment
            obs, _ = env.reset()

            # Run the model for a few steps to visualize its performance
            for _ in range(300):  # Reduced to 200 steps for faster visualization
                action, _states = self.model.predict(obs)
                result = env.step(action)

                obs, reward, terminated, truncated, info = result

                # if len(result) == 5:
                #     obs, reward, terminated, truncated, info = result
                # else:
                #     obs, reward, done, info = result
                #     terminated, truncated = done, done

                # Render the environment
                env.render()

                # Reset if episode ends
                if terminated or truncated:
                    obs, _ = env.reset()

            # Close the environment after visualization
            env.close()

        return True


    


def train(args):

    # # set max steps per episode
    # from gymnasium.wrappers import TimeLimit
    # env = make_vec_env(lambda: TimeLimit(Go1MujocoEnv(render_mode=None), max_episode_steps=1000), n_envs=1)

    vec_env = make_vec_env(
        Go1MujocoEnv,
        env_kwargs={"ctrl_type": args.ctrl_type},
        n_envs=args.num_parallel_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    training_name = "rew_uu_len_uu_net_64_64_normal_action"

    # train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    # run_name = f"{train_time}-{args.run_name}" if args.run_name else train_time

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = training_name

    model_path = f"{MODEL_DIR}/{run_name}"

    print(f"Training on {args.num_parallel_envs} parallel training environments and saving models to '{model_path}'")

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=model_path,
        log_path=LOG_DIR,
        eval_freq=args.eval_frequency,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    tensorboard_callback = TensorboardCallback()

    if args.model_path is not None:
        model = PPO.load(path=args.model_path, env=vec_env, verbose=1, tensorboard_log=LOG_DIR)
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, policy_kwargs=dict(net_arch=[64, 64]), tensorboard_log=LOG_DIR)

    visualize_callback = VisualizeCallback(check_freq=5000)

    # Combine callbacks
    callbacks = [eval_callback, visualize_callback, tensorboard_callback]

    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=run_name,
        callback=callbacks,
    )

    # Save final model
    model.save(f"{model_path}/final_model")





def test(args):

    model_path = Path(args.model_path)

    if not args.record_test_episodes:
        # Render the episodes live
        env = Go1MujocoEnv(
            ctrl_type=args.ctrl_type,
            render_mode="human",
        )
        inter_frame_sleep = 0.016
    else:
        # Record the episodes
        env = Go1MujocoEnv(
            ctrl_type=args.ctrl_type,
            render_mode="rgb_array",
            camera_name="tracking",
            width=1920,
            height=1080,
        )
        env = gym.wrappers.RecordVideo(env, video_folder="recordings/", name_prefix=model_path.parent.name)
        inter_frame_sleep = 0.0

    model = PPO.load(path=model_path, env=env, verbose=1)

    num_episodes = args.num_test_episodes
    total_reward = 0
    total_length = 0

    # user commands
    lin_x_range = [0.5, 4.0]
    n_steps_full_cycle = 600

    for _ in tqdm(range(num_episodes)):

        obs, _ = env.reset()
        
        ep_len = 0
        ep_reward = 0

        while True:
            
            # lin_x = lin_x_range[0] + (lin_x_range[1] - lin_x_range[0]) * (np.sin(2.0 * np.pi * ep_len / n_steps_full_cycle) + 1.0) / 2.0
            # lin_x = float(lin_x)
            # env._desired_velocity = np.array([lin_x, 0.0, 0.0])

            action, _ = model.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)

            env.render()

            # print("distance_from_origin", np.linalg.norm(env.data.qpos[0:2], ord=2))

            ep_reward += reward
            ep_len += 1

            # # Slow down the rendering
            time.sleep(inter_frame_sleep)

            if terminated or truncated:
                print(f"{ep_len=}  {ep_reward=}")
                break
            
            # # monitor reference tracking
            # velocity = env.data.qvel.flatten()
            # base_lin_vel = velocity[:3]
            # base_ang_vel = velocity[3:6]

            # print(base_lin_vel[0], env._desired_velocity[0], 
            #       base_lin_vel[1], env._desired_velocity[1],
            #       base_ang_vel[2], env._desired_velocity[2])

        total_length += ep_len
        total_reward += ep_reward

    # Close the environment after visualization
    env.close()
        
    print(f"Avg episode reward: {total_reward / num_episodes}, avg episode length: {total_length / num_episodes}")





if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument("--run", type=str, required=True, choices=["train", "test"])

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom name of the run. Note that all runs are saved in the 'models' directory and have the training time prefixed.",
    )

    parser.add_argument(
        "--num_parallel_envs",
        type=int,
        default=12,
        help="Number of parallel environments while training",
    )

    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=5,
        help="Number of episodes to test the model",
    )

    parser.add_argument(
        "--record_test_episodes",
        action="store_true",
        help="Whether to record the test episodes or not. If false, the episodes are rendered in the window.",
    )

    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=5_000_000,
        help="Number of timesteps to train the model for",
    )

    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=10_000,
        help="The frequency of evaluating the models while training",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model (.zip). If passed for training, the model is used as the starting point for training. If passed for testing, the model is used for inference.",
    )

    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["torque", "position"],
        default="position",
        help="Whether the model should control the robot using torque or position control.",
    )

    parser.add_argument("--seed", type=int, default=0)
    
    
    args = parser.parse_args()


    if args.run == "train":
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        train(args)
    elif args.run == "test":
        if args.model_path is None:
            raise ValueError("--model_path is required for testing")
        test(args)


# python github_train.py --run train

# python github_train.py --run test --model_path models/2025-01-29_17-53-14/best_model.zip

# provare in torque e vedere se impara
# tensorboard (display lunghezza da orgine)
# sostituire xml con quello di managerie e vedere se va ancora
        

# Note: attenzione a funzione: _set_action_space in mujoco_env.py!
        

# python github_train.py --run test --model_path models/rew_1400_len_570_net_64_64_normal_action/best_model.zip
# python github_train.py --run test --model_path models/rew_2000_len_750_net_64_64_normal_action/best_model.zip