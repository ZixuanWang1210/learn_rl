from ray import tune,air
import ray
import os
import json
# import shutil

ray.init(ignore_reinit_error=True)  # 初始化Ray

save_dir = "/workspaces/save"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


config = {
    "env": "Pusher-v4",
    "framework": "torch",  
    "num_workers": 10,
    "lr": tune.grid_search([0.01, 0.001]),  # Reduced to 2 values
    "gamma": tune.grid_search([0.99, 0.999]),  # Reduced to 2 values
    "train_batch_size": tune.grid_search([1000, 5000]),  # Reduced to 2 values
    "vf_loss_coeff": tune.grid_search([0.1, 1.0]),  # Reduced to 2 values
    "entropy_coeff": tune.grid_search([0.01, 0.1]),  # Reduced to 2 values
    "clip_param": 0.2,  # Fixed to a commonly used value
    "sgd_minibatch_size": tune.grid_search([128, 512]),  # Reduced to 2 values
    "lambda": 0.95,  # Fixed to a commonly used value
    "model": {
        "fcnet_hiddens": tune.grid_search([[256, 256], [64, 64]]),  # Reduced to 2 values
        "fcnet_activation": "relu",  # Fixed to one activation
    },
    "num_sgd_iter": tune.grid_search([3, 5]),  # Reduced to 2 values
}

analysis = tune.run(
    "PPO",
    config=config,
    stop={"training_iteration": 10},
    num_samples=3,
    fail_fast=False,
    max_failures=0,
    raise_on_failed_trial=False
)

trials = analysis.trials

N = 10

# 过滤掉未完成的试验
completed_trials = [trial for trial in trials if trial.status == "TERMINATED"]

# 按照“episode_reward_mean”排序并获取前N个已完成的试验
best_trials = sorted(completed_trials, key=lambda trial: trial.last_result.get("episode_reward_mean", 0), reverse=True)[:N]
for i, best_trial in enumerate(best_trials):
    Trainable = best_trial.get_trainable_cls()
    
    config = best_trial.config
    with open(f"{save_dir}/best_model_{i+1}_config.json", "w") as f:
        json.dump(config, f)

    best_model = Trainable(config)
    
    checkpoint_path = f"{save_dir}/best_model_{i+1}"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # else:
    #     for filename in os.listdir(checkpoint_path):
    #         file_path = os.path.join(checkpoint_path, filename)
    #         try:
    #             if os.path.isfile(file_path) or os.path.islink(file_path):
    #                 os.unlink(file_path)
    #             elif os.path.isdir(file_path):
    #                 shutil.rmtree(file_path)
    #         except Exception as e:
    #             print(f"Failed to delete {file_path}. Reason: {e}")

    print("=====> Checkpoint saved in directory :" + best_model.save_checkpoint(checkpoint_path))