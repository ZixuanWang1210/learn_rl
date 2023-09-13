from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print

config = {
    "env": "Pusher-v4",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 10,
    "lr": 0.001,
    # other parameters...
}

trainer = PPOTrainer(config=config)

for i in range(10):
    result = trainer.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = trainer.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
