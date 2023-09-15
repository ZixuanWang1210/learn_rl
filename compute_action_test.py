# Note: `gymnasium` (not `gym`) will be **the** API supported by RLlib from Ray 2.3 on.
try:
    import gymnasium as gym

    gymnasium = True
except Exception:
    import gym

    gymnasium = False

from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray import tune


# Your dictionary configuration
config = {
    "env": "Pusher-v4",
    "framework": "torch",
    "num_workers": 20,
    "lr": 0.001,
    "gamma": 0.999,
    "train_batch_size": 1000,
    "vf_loss_coeff": 0.1,
    "entropy_coeff": 0.1,
    "clip_param": 0.2,
    "sgd_minibatch_size": 128,
    "lambda": 0.95,
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu"
    },
    "num_sgd_iter": 3,
    "clip_rewards": True,
    "observation_filter": "MeanStdFilter",
    "grad_clip": 0.5,
}

# Initialize Ray and RLlib
ray.init()

# Train an agent using PPO and save the checkpoint
analysis = tune.run("PPO", config=config, stop={"training_iteration": 1}, checkpoint_at_end=True)
checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial("episode_reward_mean"))
checkpoint_path = checkpoints[0][0]


env_name = "Pusher-v4"
env = gym.make(env_name, render_mode="human")
config = PPOConfig()
(
    config
    .rollouts(num_rollout_workers=20)
    .resources(num_gpus=0)
    .environment(env="Pusher-v4")
    .framework("torch")
    .training(
        lr=0.001,
        gamma=0.999,
        train_batch_size=1000,
        vf_loss_coeff=0.1,
        entropy_coeff=0.1,
        clip_param=0.2,
        sgd_minibatch_size=128,
        lambda_=0.95,
        grad_clip=0.5
    )
)

config.fcnet_hiddens = [256, 256]  # default
config.fcnet_activation = "relu"  # default
config.clip_rewards = True  # default
config.observation_filter = "MeanStdFilter"  # default

algo=config.build()

algo.restore(checkpoint_path)

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