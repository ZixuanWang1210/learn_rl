# Note: `gymnasium` (not `gym`) will be **the** API supported by RLlib from Ray 2.3 on.
try:
    import gymnasium as gym

    gymnasium = True
except Exception:
    import gym

    gymnasium = False

from ray.rllib.algorithms.ppo import PPOConfig

env_name = "Pusher-v4"
env = gym.make(env_name,render_mode="human")
algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="Pusher-v4")
    .build()
)

algo.restore("/workspaces/save/best_model_1")

episode_reward = 0
terminated = truncated = False

if gymnasium:
    obs, info = env.reset()
else:
    obs = env.reset()

while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    print(action)
    print("----------------------------------------------")
    if gymnasium:
        obs, reward, terminated, truncated, info = env.step(action)
    else:
        obs, reward, terminated, info = env.step(action)
    episode_reward += reward