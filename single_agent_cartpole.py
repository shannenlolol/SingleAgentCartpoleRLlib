import ray
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from ray.rllib.algorithms.ppo import PPOConfig
import os
from datetime import datetime
import argparse

# Argument parser for user inputs
parser = argparse.ArgumentParser(description="Run CartPole using PPO from RLlib.")
parser.add_argument("--load", action="store_true", help="Load a pre-trained model")
parser.add_argument("--model_path", type=str, default=None, help="Path to save/load model")
parser.add_argument("--num_episodes", type=int, default=4, help="Number of evaluation episodes")
args = parser.parse_args()

# Initialize Ray
ray.init()

# Set current datetime for model and video folder naming
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = args.model_path or f"./saved_model/{current_datetime}"
video_folder = f"./video/{current_datetime}"

# Configure the PPO algorithm
config = PPOConfig()
config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size=128)
config = config.resources(num_gpus=0)
config = config.env_runners(num_env_runners=1)

# Build a PPO algorithm object from the config
algo = config.build(env="CartPole-v1")

if args.load:
    if not args.model_path:
        raise ValueError("Please provide a model path to load using the --model_path argument.")
    
    # Load the model from the provided path
    algo.restore(args.model_path)
    print(f"Model loaded from: {args.model_path}")

else:
    # Train the model
    for i in range(1000):
        result = algo.train()
        print(f"Iteration {i}: reward = {result['env_runners']['episode_reward_mean']}")

    # Save the trained model
    os.makedirs(model_save_path, exist_ok=True)
    algo.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

# Visualize and record evaluation episodes
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder, name_prefix="eval", episode_trigger=lambda e: True)

for episode_num in range(args.num_episodes):
    obs, info = env.reset()
    episode_over = False
    episode_reward = 0
    while not episode_over:
        action = algo.compute_single_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_over = done or truncated
        episode_reward += reward
    print(f"Episode {episode_num + 1}: reward = {episode_reward}")

env.close()

# Shutdown Ray
ray.shutdown()
