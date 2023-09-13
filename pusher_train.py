from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


# config = {
#     "env": "Pusher-v4",
#     "framework": "torch",  # 使用PyTorch
#     "num_gpus": 1,
#     "num_workers": 10,
#     "lr": 0.001,  # 学习率
#     "gamma": 0.999,        # 削减因子
#     "train_batch_size": 2000,  # 训练批次大小
#     "vf_loss_coeff": 0.5,  # 值函数损失系数
#     "entropy_coeff": 0.1,  # 熵正则化系数
#     "num_sgd_iter": 3,  # SGD迭代次数
#     # 其他可调参数...
# }

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=4)
    .training(gamma=0.9, lr=0.01, kl_coeff=0.3,train_batch_size=2000,vf_loss_coeff=0.5,entropy_coeff=0.1,num_sgd_iter=3)
    .resources(num_gpus=1)
    .environment(env="Pusher-v4")
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")