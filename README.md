### 目录介绍
- pusher_train.py	使用algo.train()，一次训练一组特定的参数并保存该参数下的ckp
- compute_action.py	加载使用`pusher_train.py`训练的模型，并可视化展示
- compute_action_test.py    使用指定的参数训练模型，并在训练结束后立即可视化展示

- pusher_train_with_tune_demo.py	使用tune，设置多组参数同时训练，对应的算法config和ckp。参数较少，便于测试
- pusher_train_with_tune_full.py	在demo的基础上增加参数的组数，也是实际使用的训练脚本
- compute_action_tune.py		加载使用tune训练的模型，并可视化展示

### 单机部署
描述：在devcontainer里启动head，py代码（Job）在本机运行。当devcontainer关闭时，则ray集群关闭。

**下载仓库**

```python
git clone https://e.coding.net/thu-sig-cop/jiqunhuaqianghuaxuexi/Pusher-v4.git
```
**启动devcontainer**

**启动Ray**

若此时不启动，训练开始时会自动启动，但会在训练结束后立刻自动关闭。

在容器内执行
```python
ray start --head
```
**转发和查看dashboard**
在本地主机执行

```python
ssh -L 8265:localhost:8265 xxx@10.10.114.14
```


**开始训练**

#### 使用train()方法

使用algo.train()，训练一组参数并保存ckp

```python
python pusher_train.py
```
训练模型保存在：/root/ray_results/PPO_Pusher-v4_2023-09-14_15-55-21n36c2dz6/checkpoint_000006

**可视化模型推理**

请修改代码中第23行ckp的地址
```python
python compute_action.py
```
每一步action的具体值将在终端打印

#### 使用tune
```python
pusher_train_with_tune_demo.py
```
使用tune不仅需要保存ckp还需要保存每一次实验所使用的算法config

代码中默认保存episode_reward_mean的值前6大的模型，可以修改 38 行变量N的值修改

相关文件保存在 `/workspaces/save`内，目录结构如下
```python
best_model_1              best_model_2              best_model_3              best_model_4              best_model_5              best_model_6
best_model_1_config.json  best_model_2_config.json  best_model_3_config.json  best_model_4_config.json  best_model_5_config.json  best_model_6_config.json
```
**可视化模型推理**

修改17和19行模型的地址

```python
python compute_action_tune.py
```
一步action的具体值将在终端打印


**训练监看**

```python
tensorboard --logdir=/root/ray_results
```
/root/ray_results下形如PPO_Pusher-v4_2023-09-14_12-37-28ft7zbkvh的文件夹，是algo.train()的日志文件

/root/ray_results/PPO下形如PPO_Pusher-v4_fbfec_00108_108_clip_param=0.1000,entropy_coeff=0.0050,gamma=0.9900,lambda=0.9000,lr=0.0001,fcnet_activation=tanh,fc_2023-09-14_06-37-12的文件夹，是使用tune训练的日志文件

### 集群部署
描述：在liukai13上持续运行一个Head，在liukai14上的devcontainer中创建一个子节点并添加到集群中。devcontainer同时安装有开发环境，Job从devcontainer发布到集群中。当devcontainer关闭时，liukai14也就从ray集群中退出。

备注：目前liukai13上已经运行了一个head，不必再次在liukai13上执行ray start --head

**下载仓库**
```python
git clone https://e.coding.net/thu-sig-cop/jiqunhuaqianghuaxuexi/Pusher-v4.git
```
**启动devcontainer**

**连接到Ray Master**

在容器内执行
```python
ray start --address='10.10.114.13:6379'
```
**查看dashboard**

在本地主机执行
```python
ssh -L 8265:localhost:8265 xxx@10.10.114.13
```
