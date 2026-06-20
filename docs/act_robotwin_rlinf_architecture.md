# ACT-RoboTwin 在 RLinf 中的 RL 后训练架构说明

本文解释当前项目里“旧 ACT checkpoint 如何接入 RLinf，并在 RoboTwin 里做 GRPO 后训练”的完整代码链路。读完后，你应该能回答三件事：

1. 我们为了 ACT 接入 RLinf 新增和修改了哪些文件。
2. 一次训练从 RoboTwin 观测输入，到 ACT 输出动作，再到 actor 更新参数，是如何流动的。
3. 后面调试或继续开发时，应该按什么顺序读源码。

## 1. 当前方案一句话

旧 ACT 本体仍然是原来的 RoboTwin DETR-VAE：

```text
qpos + images -> DETR-VAE ACT -> a_hat
```

这里的 `a_hat` 是旧 ACT 动作头输出的 normalized action chunk，形状大致是：

```text
[batch, 50, 14]
```

我们没有把旧 ACT 重写成另一个网络，也没有改掉它的 `action_head = Linear(hidden_dim, 14)`。真正新增的是一层 RLinf adapter：

```text
旧 ACT 输出 mean
  -> Normal(mean, std)
  -> train 时采样 normalized action
  -> eval 时直接用 mean
  -> 反归一化成 RoboTwin qpos action
  -> 记录 logprob 和 forward_inputs
  -> actor 用这些信息做 GRPO/PPO-style policy update
```

所以，核心变化不是“ACT 最后一层输出改了”，而是“ACT 外面包了一层可以被 RLinf rollout 和训练的 stochastic policy”。

## 2. 新增和改动文件清单

| 文件 | 作用 | 你重点看什么 |
| --- | --- | --- |
| `rlinf/models/embodiment/act/__init__.py` | ACT 模型包入口 | `get_model(cfg, torch_dtype)` 如何返回 `ACTRLPolicy` |
| `rlinf/models/embodiment/act/act_rl_policy.py` | ACT-RLinf adapter 主体 | 旧 ACT 加载、qpos/image 预处理、Normal 分布、`predict_action_batch`、`default_forward` |
| `examples/embodiment/config/model/act.yaml` | ACT 模型配置 | checkpoint、`dataset_stats.pkl`、旧 ACT 网络超参、`initial_logstd` |
| `examples/embodiment/config/robotwin_beat_block_hammer_eval_act.yaml` | RoboTwin + ACT 的完整运行配置 | runner、env、rollout、actor、GRPO、offload、video、batch size |
| `rlinf/config.py` | RLinf 支持模型枚举 | `SupportedModel.ACT` 和 `EMBODIED_MODEL` |
| `rlinf/models/__init__.py` | 模型注册中心 | `_build_act()` 和 `register_model("act", ...)` |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | rollout worker | 把 ACT 放进连续动作 policy 分支，调用 `predict_action_batch` |
| `rlinf/algorithms/advantages.py` | advantage 算法 | `compute_grpo_advantages` 如何按 group 做相对优势 |
| `docs/act_robotwin_rlinf_posttrain.md` | 操作 runbook | 环境准备、only_eval、GRPO 小规模训练、扩容路线 |
| `docs/act_robotwin_rlinf_architecture.md` | 本文档 | 代码架构和数据流解释 |

## 3. 推荐读代码顺序

不要一上来从 `act_rl_policy.py` 扎进去。更清楚的顺序是：

1. `examples/embodiment/config/robotwin_beat_block_hammer_eval_act.yaml`
   先看我们这次跑什么任务、用什么 actor/rollout/env、GRPO 参数是什么。

2. `examples/embodiment/train_embodied_agent.py`
   看 Hydra config 如何进入 RLinf，如何创建 Cluster、Actor、Rollout、Env worker。

3. `rlinf/runners/embodied_runner.py`
   看主训练循环：同步权重、生成 rollout、收轨迹、算 advantage、actor update、eval、save checkpoint。

4. `rlinf/workers/rollout/hf/huggingface_worker.py`
   看 rollout worker 如何接收 env obs，调用 ACT 预测动作，把动作和 logprob 发回 env。

5. `rlinf/models/embodiment/act/act_rl_policy.py`
   看 ACT adapter 的细节：输入怎么处理、输出怎么采样、logprob 怎么算。

6. `rlinf/workers/env/env_worker.py`
   看 env worker 如何发送观测、接收 action chunk、组装 trajectory。

7. `rlinf/envs/robotwin/robotwin_env.py`
   看 RoboTwin 原始 obs 如何变成 RLinf obs，action chunk 如何真正送进 RoboTwin。

8. `rlinf/data/embodied_io_struct.py`
   看 `EnvOutput`、`RolloutResult`、`Trajectory` 这些容器如何保存数据。

9. `rlinf/workers/actor/fsdp_actor_worker.py`
   看 actor 如何接收 trajectory、算 advantage、重算 logprob、反向传播更新模型。

10. `rlinf/algorithms/utils.py`、`rlinf/algorithms/advantages.py`、`rlinf/algorithms/losses.py`
    看 GRPO advantage 和 PPO-style clipped policy loss 的数学实现。

## 4. 运行时整体架构

启动入口是：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/lhj/RLinf-main/examples/embodiment/config \
  --config-name robotwin_beat_block_hammer_eval_act \
  runner.only_eval=False ...
```

`train_embodied_agent.py` 会创建三个主要 worker group：

```text
                    Ray Cluster
                         |
        +----------------+----------------+
        |                |                |
   Actor Worker     Rollout Worker     Env Worker
   FSDP 训练 ACT     ACT 推理采动作     RoboTwin 环境
        |                |                |
        |  sync weights  |                |
        +--------------->|                |
        |                |                |
        |                | env obs        |
        |                |<---------------+
        |                |                |
        |                | actions + logprob
        |                +--------------->|
        |                |                |
        | trajectory batch                |
        |<--------------------------------+
        |
   advantage + loss + backward + optimizer step
```

注意：Actor 和 Rollout 里都有一份 ACT 模型。

- Actor 负责训练和保存 checkpoint。
- Rollout 负责用当前策略和环境交互。
- 每个训练 step 开始时，Actor 把最新权重同步给 Rollout。

这也是为什么配置里有：

```yaml
rollout:
  enable_offload: True

actor:
  enable_offload: True
```

在 8GB 显存上，actor/rollout 都常驻 GPU 会很容易 OOM，所以我们用 offload 降低显存峰值。

## 5. 配置层：这次到底训练什么

主配置在：

```text
examples/embodiment/config/robotwin_beat_block_hammer_eval_act.yaml
```

关键字段：

```yaml
defaults:
  - env/robotwin_beat_block_hammer@env.train
  - env/robotwin_beat_block_hammer@env.eval
  - model/act@actor.model
```

这表示：

- train env 和 eval env 都用 RoboTwin 的 `beat_block_hammer` 任务。
- actor model 用 `examples/embodiment/config/model/act.yaml`。
- rollout model 从 actor model 镜像一份配置。

ACT 模型配置在：

```text
examples/embodiment/config/model/act.yaml
```

关键字段：

```yaml
model_type: "act"
model_path: "/home/lhj/robot_l/robotwin/policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50/policy_last.ckpt"
dataset_stats_path: "/home/lhj/robot_l/robotwin/policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50/dataset_stats.pkl"
act_source_dir: "/home/lhj/robot_l/robotwin/policy/ACT"

camera_names: ["cam_high", "cam_left_wrist", "cam_right_wrist"]
state_dim: 14
action_dim: 14
num_action_chunks: 50

image_normalization: "imagenet"
image_resize_hw: [480, 640]
eval_stepwise: true
temporal_agg: true
temporal_agg_decay: 0.01
initial_logstd: -2.0
min_logstd: -5.0
max_logstd: 1.0
use_stochastic_eval: false
```

这些字段的意义：

- `model_path`：旧 ACT imitation learning checkpoint。这里用 `policy_last.ckpt`，
  因为旧 `policy/ACT/eval.sh` / `deploy_policy.py` 的 baseline 路径实际加载
  `policy_last.ckpt`。
- `dataset_stats_path`：旧 ACT 训练时保存的数据均值方差，用来 normalize qpos、denormalize action。
- `act_source_dir`：旧 RoboTwin ACT 源码目录，adapter 从这里 import `detr.models.build_ACT_model`。
- `camera_names`：ACT eval 输入相机顺序。这里对齐旧 `deploy_policy.py` 的实际输入：
  头部相机、左腕相机、右腕相机。
- `num_action_chunks: 50`：ACT 一次预测 50 步动作。
- `image_normalization: "imagenet"`：旧 ACT deployment wrapper 先把图像缩放到 `[0, 1]`，再做 ImageNet mean/std normalization。
- `image_resize_hw: [480, 640]`：旧 `deploy_policy.py` 会把 RoboTwin
  相机图像从环境返回的 `240x320` resize 到 `480x640` 再输入 ACT。这个
  resize 缺失时，qpos 对齐但 action parity 会明显偏掉。
- `eval_stepwise: true`：ACT baseline eval 时每次只向环境发送 1 步动作，用来对齐旧 RoboTwin `take_action` 评测语义。
- `temporal_agg: true`：stepwise eval 时复用旧 ACT 的 temporal aggregation 思路，每个 sim step 重新预测 chunk 并聚合覆盖当前步的历史预测。
- `initial_logstd: -2.0`：RL adapter 新增的可学习动作标准差初始值。

## 6. 模型注册：RLinf 怎么认识 act

RLinf 原来不知道 `model_type: act` 是什么，所以我们做了三层注册。

第一层：`rlinf/config.py`

```python
SupportedModel.ACT = SupportedModel.register("act", force=True)
EMBODIED_MODEL.add(SupportedModel.ACT)
```

作用：让 Hydra validate config 时接受 `act`，并确认它是 embodied model。

第二层：`rlinf/models/__init__.py`

```python
def _build_act(cfg, torch_dtype):
    from rlinf.models.embodiment.act import get_model
    return get_model(cfg, torch_dtype)

register_model(SupportedModel.ACT.value, _build_act, category="embodied", force=True)
```

作用：RLinf 调用 `get_model(cfg)` 时，可以根据 `cfg.model_type == "act"` 找到 ACT builder。

第三层：`rlinf/models/embodiment/act/__init__.py`

```python
from .act_rl_policy import ACTRLPolicy

def get_model(cfg, torch_dtype):
    return ACTRLPolicy(cfg, torch_dtype=torch_dtype)
```

作用：真正构造我们的 ACT adapter。

## 7. ACT adapter：旧 ACT 如何变成 RL policy

核心文件：

```text
rlinf/models/embodiment/act/act_rl_policy.py
```

核心类：

```python
class ACTRLPolicy(nn.Module, BasePolicy):
```

它有两个身份：

- `nn.Module`：PyTorch 可以训练它的参数。
- `BasePolicy`：RLinf rollout/actor worker 知道怎么调用它。

### 7.1 初始化阶段

初始化时做这些事：

```text
读取 cfg
  -> 找到旧 ACT 源码 act_source_dir
  -> import build_ACT_model
  -> 用旧 ACT 超参创建 DETR-VAE
  -> 加载旧 checkpoint
  -> 加载 dataset_stats.pkl
  -> 注册 action/qpos mean/std buffer
  -> 新增可学习 logstd 参数
```

这里的 `logstd` 是这次 RL adapter 新增的：

```python
self.logstd = nn.Parameter(torch.full((1, 50, 14), initial_logstd))
```

它不是旧 ACT checkpoint 里来的。旧 ACT 只给动作预测值，RL 需要一个概率分布，所以我们要有 std：

```text
std = exp(logstd)
```

### 7.2 Rollout 前向：predict_action_batch

rollout worker 调用：

```python
actions, result = self.hf_model.predict_action_batch(env_obs=env_obs, mode="train")
```

ACT adapter 内部数据流：

```text
env_obs["states"]
  -> 截取前 14 维
  -> (states - qpos_mean) / qpos_std
  -> normalized qpos

env_obs["main_images"] + env_obs["wrist_images"]
  -> [B, H, W, C] / [B, 2, H, W, C]
  -> [B, 3 cameras, 3, H, W]
  -> ImageNet normalize

normalized qpos + images
  -> old ACT DETR-VAE
  -> a_hat / mean: [B, 50, 14]

mean + logstd
  -> Normal(mean, exp(logstd))

train:
  normalized_action ~ Normal(mean, std)

eval:
  normalized_action = mean

normalized_action
  -> normalized_action * action_std + action_mean
  -> env_action: [B, 50, 14]
```

这里要注意两个动作空间：

| 名字 | 空间 | 用途 |
| --- | --- | --- |
| `normalized_action` | 旧 ACT 训练数据的 normalized 空间 | 计算 logprob、训练 policy |
| `env_action` | RoboTwin qpos 真实量纲 | 送进 RoboTwin 环境执行 |

### 7.3 为什么要保存 forward_inputs

rollout 时，adapter 返回：

```python
result = {
    "prev_logprobs": logprobs,
    "prev_values": values,
    "forward_inputs": {
        "qpos": qpos,
        "images": images,
        "action": normalized_actions,
    },
}
```

这些字段分别是什么意思：

| 字段 | 人话解释 | 为什么要保存 |
| --- | --- | --- |
| `qpos` | 当时环境里的 normalized robot state | actor 更新时要复现当时 ACT 看见的输入 |
| `images` | 当时环境里的三路相机图像 | actor 更新时重算当前策略概率 |
| `action` | 当时采样出来的 normalized action | actor 要问“现在的策略对这个动作给多大概率” |
| `prev_logprobs` | rollout 当时旧策略对这个动作的 log probability | PPO/GRPO ratio 的分母 |
| `prev_values` | critic value | ACT 当前没有 value head，所以这里是 0 |

`forward_inputs` 这个名字的意思就是：“以后再做一次 forward 所需要的输入”。它不是环境输入的全部，只保存训练重算 logprob 必须用到的东西。

### 7.4 训练前向：default_forward

actor 更新时调用：

```python
output_dict = self.model(
    forward_inputs=forward_inputs,
    compute_logprobs=True,
)
```

这会进入 ACT adapter 的：

```python
default_forward(...)
```

它做的事情是：

```text
读取 forward_inputs["qpos"]
读取 forward_inputs["images"]
读取 forward_inputs["action"]
  -> 用当前 ACT 参数重新预测 mean
  -> 用当前 logstd 重新构造 Normal(mean, std)
  -> 计算当前策略对旧 action 的 logprob
```

所以：

```text
prev_logprobs = rollout 时旧策略认为这个动作的概率
logprobs      = update 时当前策略认为同一个动作的概率
ratio         = exp(logprobs - prev_logprobs)
```

这就是 PPO/GRPO 更新需要的核心量。

## 8. RoboTwin 环境数据流

核心文件：

```text
rlinf/envs/robotwin/robotwin_env.py
```

RoboTwin 原始观测大致包含：

```text
full_image
left_wrist_image
right_wrist_image
state
instruction
```

`RoboTwinEnv._extract_obs_image()` 把它转换成 RLinf 统一格式：

```python
extracted_obs = {
    "main_images": batch_images,
    "wrist_images": batch_wrist_images,
    "states": batch_states,
    "task_descriptions": batch_instructions,
}
```

ACT adapter 主要用：

- `states`
- `main_images`
- `wrist_images`

当前 ACT 不使用 `task_descriptions`。它是视觉 + qpos policy，不是语言 VLA。

### 8.1 action chunk 如何进环境

训练和普通 rollout 中，ACT 一次输出：

```text
[num_envs, 50, 14]
```

`RoboTwinEnv.chunk_step(chunk_actions)` 直接把这个 chunk 送给 RoboTwin vector env：

```python
raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(chunk_actions)
```

也就是说，训练路径仍然是：

```text
观察一次
  -> ACT 预测 50 步
  -> 环境执行这 50 步
  -> 再拿新观察
  -> 再预测下一个 50 步
```

为了对齐旧 RoboTwin ACT baseline eval，当前配置额外支持 stepwise eval：

```text
每个 sim step
  -> ACT 重新预测 50-step chunk
  -> temporal aggregation 聚合覆盖当前步的历史预测
  -> 只把 1 步动作送进环境
  -> RoboTwin 通过 take_action(action) 执行这一小步
```

ACT eval 配置里对应 `env.eval.step_mode: take_action`。这个开关只用于
baseline parity：它让 RLinf 走旧 RoboTwin ACT deploy 的逐步执行语义，而
训练路径仍然保持 `gen_sparse_reward_data(chunk_actions)` 的 chunk reward
收集方式。

### 8.2 reward 怎么来

当前配置里 reward model 是关闭的：

```yaml
reward:
  use_reward_model: False
```

所以 reward 主要来自 RoboTwin 环境本身。`RoboTwinEnv._calc_step_reward()` 里，如果使用 custom reward：

```text
reward = reward_coef * terminations
```

也就是成功终止时给奖励。配置里还有：

```yaml
env.train.ignore_terminations: True
```

这表示成功信号会被记录到 metrics 里，但训练环境不会因为成功而提前结束整段 rollout。这样更适合固定长度 chunk rollout。

## 9. Rollout worker：怎么生成一段轨迹

核心文件：

```text
rlinf/workers/rollout/hf/huggingface_worker.py
```

关键函数：

- `predict()`
- `generate_one_epoch()`
- `generate()`
- `sync_model_from_actor()`

rollout worker 每个 chunk step 做：

```text
从 env_channel 收 env obs
  -> 调 ACT.predict_action_batch()
  -> 得到 actions + prev_logprobs + forward_inputs
  -> 包成 RolloutResult
  -> 发回 env_channel
```

`RolloutResult` 里保存：

```python
RolloutResult(
    actions=actions,
    prev_logprobs=result["prev_logprobs"],
    prev_values=result["prev_values"],
    bootstrap_values=...,
    forward_inputs=result["forward_inputs"],
    versions=current_weight_version,
)
```

`versions` 是当前 rollout 用的是哪一版 actor 权重。普通同步训练里它主要用于记录和调试；异步或 decoupled PPO 时更重要。

## 10. Env worker：怎么把交互变成 trajectory

核心文件：

```text
rlinf/workers/env/env_worker.py
```

env worker 的主函数是：

```python
interact(...)
```

内部大致做：

```text
bootstrap reset，拿初始 obs
  -> send_env_batch() 把 obs 发给 rollout

循环 n_train_chunk_steps:
  -> recv_rollout_results() 从 rollout 收 action chunk
  -> env_train_step() / chunk_step() 执行动作
  -> 得到 next obs、reward、done、info
  -> 把当前 step 的 action/logprob/reward/done/forward_inputs 存入 EmbodiedRolloutResult
  -> 把 next obs 再发给 rollout

rollout 结束:
  -> EmbodiedRolloutResult.to_trajectory()
  -> send_rollout_trajectories() 发给 actor
```

这里有一条容器转换链：

```text
EnvOutput
  表示环境某个时刻的 obs/reward/done

RolloutResult
  表示 rollout policy 对这个 obs 产生的 action/logprob/forward_inputs

ChunkStepResult
  表示一个 chunk step 中 model output + env output 的合并结果

EmbodiedRolloutResult
  在 env worker 内部逐步 append，收集整段 rollout

Trajectory
  整段 rollout 的 tensor 化结果，发给 actor

batch dict
  actor 收到多个 Trajectory 后 cat 成训练 batch
```

对应代码主要在：

```text
rlinf/data/embodied_io_struct.py
```

## 11. Actor worker：真正怎么更新 ACT

核心文件：

```text
rlinf/workers/actor/fsdp_actor_worker.py
```

actor 训练分四步。

### 11.1 接收轨迹

```python
recv_rollout_trajectories(...)
```

做：

```text
从 actor_channel 收 Trajectory
  -> convert_trajectories_to_batch()
  -> _process_received_rollout_batch()
```

原始 trajectory shape 大致是：

```text
[rollout_epoch * n_chunk_steps, batch, ...]
```

actor 会整理成更适合算 advantage 的形状：

```text
[n_chunk_steps, rollout_epoch * batch, ...]
```

### 11.2 计算 GRPO advantage

```python
compute_advantages_and_returns()
```

调用统一入口：

```python
calculate_adv_and_returns(...)
```

对 embodied task，实际链路是：

```text
preprocess_embodied_advantages_inputs()
  -> 把 chunk rewards 展平成时间序列

calculate_scores()
  -> 对每条 trajectory 累计 reward，得到 score

compute_grpo_advantages()
  -> 按 group_size 分组
  -> 每组内部 reward 做 mean/std normalize
  -> 得到相对优势 advantage

postprocess_embodied_advantages_outputs()
  -> 把 advantage 还原回 chunk 结构
```

这就是“怎么知道动作好还是坏”：

```text
不是单独看某个动作像不像 demo
而是看这一条 rollout 在同组样本里 score 高不高

score 高于组平均 -> advantage > 0
score 低于组平均 -> advantage < 0
```

当前配置里：

```yaml
algorithm:
  adv_type: grpo
  loss_type: actor
  group_size: 4   # 训练时通常 override
```

`group_size` 的意思是：同一个“问题/初始条件组”下采多条 rollout，做组内相对比较。我们在小规模 smoke run 里会用 2 或 4。

### 11.3 重算当前 logprob

actor 训练 micro batch 时：

```python
train_micro_batch(...)
```

会取出：

```python
advantages = micro_batch["advantages"]
prev_logprobs = micro_batch["prev_logprobs"]
forward_inputs = micro_batch["forward_inputs"]
```

然后调用当前 actor 模型：

```python
output_dict = self.model(
    forward_inputs=forward_inputs,
    compute_logprobs=True,
)
```

对 ACT 来说，这会进入 `ACTRLPolicy.default_forward()`：

```text
当前 ACT 参数 + 当前 logstd
  -> 对 rollout 当时采样出来的 normalized action 重新计算 logprob
```

因此：

```text
old_logprobs = prev_logprobs
logprobs     = output_dict["logprobs"]
```

### 11.4 算 loss 并反向传播

loss 入口：

```python
policy_loss(**loss_kwargs)
```

对 embodied task，先进入：

```python
preprocess_loss_inputs(...)
```

当前配置是：

```yaml
reward_type: action_level
logprob_type: action_level
```

ACT adapter 原始 logprob 是每个 action dim 一个：

```text
[B, 50 * 14]
```

`action_level` 会把每个 14 维关节动作的 logprob 求和：

```text
[B, 50, 14] -> sum over action_dim -> [B, 50]
```

这样就能和每个 chunk step 的 advantage 对齐。

实际 actor loss 在：

```text
rlinf/algorithms/losses.py
```

核心公式是 PPO-style clipped policy loss：

```text
ratio = exp(logprobs - old_logprobs)

policy_loss1 = -advantages * ratio
policy_loss2 = -advantages * clipped_ratio
policy_loss  = max(policy_loss1, policy_loss2)
```

为什么前面有负号：

```text
优化器永远在最小化 loss

advantage > 0:
  我们希望这个动作概率变大
  ratio 变大时 -advantage * ratio 变小
  最小化 loss 会推着 ratio 变大

advantage < 0:
  我们希望这个动作概率变小
  ratio 变小时 -advantage * ratio 也变小
  最小化 loss 会推着 ratio 变小
```

最后：

```text
loss.backward()
optimizer.step()
```

会更新：

- 旧 ACT DETR-VAE 的参数。
- 新增的 `logstd` 参数。

当前没有 critic/value head，所以不是 actor-critic，而是 critic-free 的 GRPO actor update。

## 12. 一次完整训练 step 的时序

下面是一轮 `runner.run()` 里的真实顺序：

```text
1. actor.set_global_step(step)
2. rollout.set_global_step(step)

3. 如果需要同步权重:
   actor.sync_model_to_rollout()
   rollout.sync_model_from_actor()

4. env.interact() 开始环境交互
5. rollout.generate() 开始 policy 推理

6. env 发送初始 obs 给 rollout
7. rollout 用 ACT 预测 action chunk
8. env 执行 action chunk，得到 reward/done/next_obs
9. 重复 6-8，直到这一轮 rollout 收集完

10. env 把 Trajectory 发给 actor
11. actor.recv_rollout_trajectories()
12. actor.compute_advantages_and_returns()
13. actor.run_training()

14. runner.global_step += 1
15. 按 val_check_interval 做 eval
16. 按 save_interval 保存 checkpoint
17. 打印并记录 metrics
```

这个 loop 的主控制在：

```text
rlinf/runners/embodied_runner.py
```

## 13. 关键变量词典

| 名字 | 在哪里出现 | 意义 |
| --- | --- | --- |
| `env_obs` | rollout worker -> ACT adapter | 当前环境观测 |
| `states` | RoboTwin obs | 机器人 qpos/state |
| `main_images` | RoboTwin obs | 头部相机图像 |
| `wrist_images` | RoboTwin obs | 左右腕部相机图像 |
| `a_hat` / `mean` | ACT adapter | 旧 ACT 输出的 normalized action mean |
| `logstd` | ACT adapter | 新增可学习探索噪声参数 |
| `normalized_action` | ACT adapter | 用于 logprob 和训练的 normalized 动作 |
| `env_action` | ACT adapter -> RoboTwin | 反归一化后的真实 qpos action |
| `prev_logprobs` | rollout 保存 | rollout 当时旧策略对 action 的 log probability |
| `logprobs` | actor update 重算 | 当前策略对同一个 action 的 log probability |
| `forward_inputs` | rollout 保存，actor 使用 | 重算 logprob 所需的 qpos/images/action |
| `advantages` | actor 计算 | 这条 rollout 相对同组结果是好还是坏 |
| `ratio` | loss 里计算 | 当前策略概率 / 旧策略概率 |
| `loss_mask` | actor/env 数据处理 | 哪些 timestep 应该参与 loss |
| `versions` | rollout 保存 | action 是由哪一版模型权重生成的 |
| `checkpoint` | runner 保存 | actor 训练后的模型权重 |

## 14. 输入和输出分别是什么

### 输入

训练启动时的外部输入：

- 旧 ACT checkpoint：`policy_last.ckpt`
- 旧 ACT stats：`dataset_stats.pkl`
- RoboTwin assets：`/home/lhj/RoboTwin-RLinf_support`
- RoboTwin task config：`beat_block_hammer`
- RLinf config：actor/rollout/env/algorithm/batch/offload 等

训练中每一步的模型输入：

```text
qpos:   [B, 14]
images: [B, 3, 3, H, W]
```

### 输出

rollout 给环境的输出：

```text
env_action: [B, 50, 14]
```

actor 训练时的输出：

```text
logprobs: [B, 50]   # action_level 后
loss: scalar
gradients: ACT params + logstd
```

训练最终输出：

```text
../results/<experiment_name>/checkpoints/global_step_<N>/actor/
```

这里保存的是 actor checkpoint。后面 eval 或 resume 应该从这个目录恢复。

## 15. 日志和指标怎么看

常见指标：

| 指标 | 意义 |
| --- | --- |
| `env/success_once` | rollout 中是否曾经成功 |
| `env/success_at_end` | episode 结束时是否成功 |
| `env/return` | 环境累计 reward |
| `actor/policy_loss` | policy gradient loss |
| `actor/ratio` | 当前策略概率与旧策略概率的比值 |
| `actor/ratio_abs` | ratio 偏离 1 的程度 |
| `actor/approx_kl` | 当前策略相对旧策略变化幅度 |
| `actor/clip_fraction` | PPO clipping 生效比例 |
| `actor/grad_norm` | 梯度范数 |
| `actor/lr` | 学习率 |

如果 `success_once` 提升，同时 `approx_kl` 不爆、`ratio_abs` 不极端、`grad_norm` 不长期异常，就说明小规模后训练方向大致健康。

## 16. 当前实现边界

当前 ACT-RLinf adapter 是一个最小可训练方案，有几个边界要明确：

1. 训练路径仍是 chunk rollout。
   ACT 一次预测 50 步，环境执行 50 步，再预测下一段。为了对齐旧 ACT baseline，eval 可以通过 `eval_stepwise: true` 和 `temporal_agg: true` 走逐步聚合。

2. 没有 critic/value head。
   `prev_values` 当前是 0，GRPO 不依赖 critic。

3. 训练时 stochastic，eval 时 deterministic。
   `train` 用 `Normal(mean, std)` 采样，`eval` 默认直接用 `mean`。

4. `logstd` 是新参数。
   它不来自旧 checkpoint，会和 ACT 一起被 RL 更新。

5. 旧 ACT checkpoint 和 RoboTwin 环境动作量纲靠 `dataset_stats.pkl` 对齐。
   qpos 输入要 normalize，action 输出要 denormalize。

6. 8GB 显存下需要保守启动。
   建议 `actor.enable_offload=True`、`rollout.enable_offload=True`、先用 2-4 个 env、video off。

## 17. 最小训练理解版数据流

如果只记一张图，记这张：

```text
RoboTwin raw obs
  full_image / wrist_images / state
        |
        v
RoboTwinEnv._extract_obs_image()
  main_images / wrist_images / states
        |
        v
RolloutWorker.predict()
        |
        v
ACTRLPolicy.predict_action_batch()
  states -> qpos normalize
  images -> camera tensor normalize
  old ACT -> normalized action mean
  Normal(mean, exp(logstd))
  sample normalized action
  denormalize -> env action
  save prev_logprobs + forward_inputs
        |
        v
RoboTwinEnv.chunk_step(env_action)
  reward / done / next obs
        |
        v
EnvWorker builds Trajectory
        |
        v
Actor.recv_rollout_trajectories()
        |
        v
Actor.compute_advantages_and_returns()
  rewards -> scores -> GRPO advantage
        |
        v
Actor.train_micro_batch()
  forward_inputs -> ACTRLPolicy.default_forward()
  recompute current logprobs
  ratio = exp(current - previous)
  loss = PPO-style clipped GRPO actor loss
        |
        v
backward + optimizer.step()
  update ACT params + logstd
        |
        v
sync updated actor weights to rollout
```

## 18. 后续开发建议

如果继续做专业化后训练，我建议按这个顺序推进：

1. 固定 clean eval baseline。
   用 `only_eval: True` 跑旧 `demo_clean-50` checkpoint，记录成功率、耗时、显存。

2. 小规模 GRPO smoke run。
   2-4 env、10-20 step、video off、offload on，只确认能真实 backward/update。

3. 小规模有效性验证。
   4 env、group_size 4、50-100 step，看 eval success 是否持平或提升。

4. 扩大 env 数量。
   先 8，再 16。每次只改一个维度，避免不知道是谁带来的变化。

5. 再开 domain randomization。
   clean 环境能稳定提升后，再逐项打开随机背景、桌面扰动、光照扰动。

6. 保存每个阶段的 run record。
   记录 checkpoint、config overrides、eval success、GPU 显存、是否 OOM。

这样做的目的不是慢，而是让每一次训练结果都能解释：到底是 ACT 本身提升了，还是环境随机性、batch size、采样 std、reward 设置改变导致的。
