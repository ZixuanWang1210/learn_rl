from ray import tune
from ray.rllib import train

# 定义配置空间
config = {
    "env": "Pusher-v4",
    "framework": "torch",  # 使用PyTorch
    "num_gpus": 1,
    "num_workers": 10,
    "lr": tune.grid_search([0.01, 0.001]),  # 学习率
    "gamma": tune.grid_search([0.99, 0.999]),        # 削减因子
    "train_batch_size": 2000,  # 训练批次大小
    "vf_loss_coeff": tune.grid_search([0.5, 1.0]),  # 值函数损失系数
    "entropy_coeff": tune.grid_search([0.01, 0.1]),  # 熵正则化系数
    "num_sgd_iter": 3,  # SGD迭代次数
    # 其他可调参数...
}

# 启动超参数搜索
analysis = tune.run(
    "PPO",                 # 使用PPO算法
    config=config,         # 配置信息
    stop={"training_iteration": 10},  # 停止条件（可选）
    num_samples=5         # 运行多少次试验（可选）
)

# 获取所有试验
trials = analysis.trials

# 获取性能最好的试验
best_trial = analysis.get_best_trial("episode_reward_mean")

# 获取训练的类和配置
Trainable = best_trial.get_trainable_cls()
config = best_trial.config

# 重新创建并加载模型
best_model = Trainable(config)
best_model.restore(best_trial.checkpoint.value)

# 保存模型
best_model.save("/root/workspaces/ray/save/best_model")
