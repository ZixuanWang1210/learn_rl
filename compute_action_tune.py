try:
    import gymnasium as gym

    gymnasium = True
except Exception:
    import gym

    gymnasium = False

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO as Trainable
import json

ray.init(ignore_reinit_error=True)

CHECKPOINT_PATH = "/workspaces/save/best_model_8"

with open("/workspaces/save/best_model_1_config.json", "r") as f:
    config = json.load(f)

best_model = Trainable(config=config)
best_model.load_checkpoint(CHECKPOINT_PATH)

env = gym.make("Pusher-v4",render_mode="human")
state = env.reset()

episode_reward = 0
terminated = truncated = False

if gymnasium:
    obs, info = env.reset()
else:
    obs = env.reset()

while not terminated and not truncated:
    action = best_model.compute_single_action(obs)
    print(action)
    print("----------------------------------------------")
    if gymnasium:
        obs, reward, terminated, truncated, info = env.step(action)
    else:
        obs, reward, terminated, info = env.step(action)
    episode_reward += reward

env.close()