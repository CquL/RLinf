# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# RLinf 具身智能训练入口脚本
# ============================================================================
# 本文件是 RLinf 框架的核心启动入口，负责将配置文件解析、集群管理、
# 各组件（Actor/Rollout/Env）的创建与启动串联为完整的训练/评估流程。
#
# ============================================================================
# 整体架构概览（8 步启动流程）
# ============================================================================
#
#   ┌──────────┐    ┌──────────┐    ┌──────────┐
#   │ 1. 配置   │───▶│ 2. 集群   │───▶│ 3. Actor │
#   │ 验证     │    │ 创建     │    │ 创建     │
#   └──────────┘    └──────────┘    └──────────┘
#                                        │
#                    ┌───────────────────┼───────────────────┐
#                    ▼                   ▼                   ▼
#              ┌──────────┐       ┌──────────┐       ┌──────────┐
#              │ 4.Rollout│       │ 5. Env   │       │ (可选)    │
#              │ 创建     │       │ 创建     │       │ Reward   │
#              └──────────┘       └──────────┘       └──────────┘
#                    │                   │                   │
#                    └───────────────────┼───────────────────┘
#                                        ▼
#                                 ┌──────────┐
#                                 │ 6.Runner │
#                                 │ 组装    │
#                                 └──────────┘
#                                   │       │
#                            ┌──────┘       └──────┐
#                            ▼                     ▼
#                     ┌──────────┐          ┌──────────┐
#                     │ 7. init  │          │ 8. run() │
#                     │ workers  │─────────▶│ 主循环   │
#                     └──────────┘          └──────────┘
#
# 运行时的数据流（训练循环）：
#
#   Env ──(obs)──▶ Rollout ──(actions)──▶ Env ──(rewards,dones)──▶ Actor ──(weights)──▶ Rollout
#    ▲                   ▲                                              │                   │
#    │                   │                                              ▼                   │
#    │                   └──────────────────────── (weights sync) ◀─────┘                   │
#    │                                                                                       │
#    └────────────────────────── (env step) ◀────────────────────────────────────────────────┘
#
# 关键设计思想：
#   1. 组件解耦：Actor(训练)、Rollout(推理/数据收集)、Env(仿真) 是独立的分布式进程
#   2. 通道通信：组件之间通过 Channel（基于 Ray 的分布式队列）异步传递数据
#   3. 权重同步：Actor 定期将更新后的模型权重推送给 Rollout，保证策略一致
#   4. 灵活部署：各组件可部署在同一节点（collocated）或不同节点（disaggregated）
#   5. FSDP 分布式训练：Actor 可利用多 GPU 进行模型分片训练
# ============================================================================

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

# 配置验证函数：校验 + 补全用户配置
from rlinf.config import validate_cfg

# EmbodiedRunner：具身智能训练/评估的主控制器，拥有整个训练循环
from rlinf.runners.embodied_runner import EmbodiedRunner

# Cluster：Ray 集群的管理抽象，负责节点的发现、资源调度和 Worker 启动
from rlinf.scheduler import Cluster

# HybridComponentPlacement：组件放置策略，决定 Actor/Rollout/Env 部署在哪些 GPU/节点上
from rlinf.utils.placement import HybridComponentPlacement

# EnvWorker：环境 Worker，负责管理仿真环境实例（如 MuJoCo、IsaacSim、RoboTwin）
from rlinf.workers.env.env_worker import EnvWorker

# EmbodiedRewardWorker：奖励模型 Worker，用于独立的 reward model 计算奖励
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

# MultiStepRolloutWorker：多步推演 Worker，负责用当前策略在环境中收集交互轨迹
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

# 设置 PyTorch 多进程启动方式为 "spawn"
# "spawn" 会在每次创建子进程时重新启动 Python 解释器，确保干净的子进程状态
# 这对 CUDA 上下文隔离至关重要（每个 GPU 进程需要独立的 CUDA 上下文）
mp.set_start_method("spawn", force=True)


# ============================================================================
# Hydra 主入口装饰器
# ============================================================================
# @hydra.main 将 main() 函数注册为 Hydra 应用入口点。
# - version_base="1.1"：Hydra 1.1 版本兼容模式
# - config_path="config"：配置文件搜索路径（相对于本脚本所在目录的 config/ 子目录）
# - config_name="maniskill_ppo_openvlaoft"：默认使用的配置文件名（不含 .yaml 后缀）
#   实际运行时可通过命令行参数 --config-name 覆盖，例如：
#   python train_embodied_agent.py --config-name robotwin_beat_block_hammer_eval_act
# ============================================================================
@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    # =========================================================================
    # 步骤 1: validate_cfg(cfg) — 配置验证与补全
    # =========================================================================
    # 为什么需要这一步？
    # 用户在 YAML 中可能只提供了部分参数或使用了占位符（如 ${oc.env:XXX}）。
    # validate_cfg() 执行以下关键操作：
    #   a) 解析所有 OmegaConf 变量插值（如 ${actor.model.precision} → "bf16"）
    #   b) 校验必填字段是否存在（如 env.train.env_type 是否为已注册的类型）
    #   c) 为缺失的可选字段填充默认值（如未指定 clip_ratio 时自动设定）
    #   d) 校验参数合法性（如 learning_rate > 0、num_envs >= 1）
    #   e) 类型检查和转换（如字符串 → 枚举类型）
    #   f) 交叉字段一致性校验（如 loss_type="actor_critic" 必须有 value_head）
    #
    # 如果不做这一步，后续 Worker 初始化时可能因配置不完整/不合法而崩溃，
    # 且错误信息难以定位。在启动最早期做集中校验是"fail-fast"原则的体现。
    # =========================================================================
    cfg = validate_cfg(cfg)

    # 打印解析后的完整配置（JSON 格式），方便调试和记录本次运行的参数
    # OmegaConf.to_container(cfg, resolve=True) 会将所有变量插值解析为最终值
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    # =========================================================================
    # 步骤 2: 创建 Cluster — 计算集群抽象
    # =========================================================================
    # 为什么需要这一步？
    # RLinf 的架构建立在 Ray 分布式框架之上。Cluster 对象承担以下角色：
    #
    #   a) Ray 集群连接管理：连接到已有的 Ray 集群（通过 ray.init() 或
    #      自动检测现有的 ray 运行时），管理节点列表和 GPU 资源
    #   b) 节点资源感知：查询每个节点的 CPU/GPU 数量、内存、网络拓扑
    #   c) Worker 生命周期管理：作为工厂（Factory），负责在各个节点上
    #      启动 Actor/Rollout/Env Worker 的 Ray Actor 实例
    #   d) 日志和监控基础设施：配置分布式日志收集路径
    #
    # 参数：
    #   cluster_cfg=cfg.cluster：从 YAML 的 cluster 段读取配置（num_nodes,
    #     component_placement 等）
    #   distributed_log_dir：每个 Worker 的独立日志存储路径
    # =========================================================================
    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )

    # =========================================================================
    # 组件放置策略
    # =========================================================================
    # HybridComponentPlacement 解析 YAML 中的 component_placement 配置
    # （如 "actor, env, rollout: 0" 表示全部放节点 0），为每个组件生成
    # 具体的放置策略对象。放置策略决定了：
    #   - 组件部署在哪些节点的哪些 GPU 上
    #   - 是否启用模型并行（tensor parallel / pipeline parallel）
    #   - 同节点 vs 跨节点通信模式
    # =========================================================================
    component_placement = HybridComponentPlacement(cfg, cluster)

    # =========================================================================
    # 步骤 3: 创建 Actor Worker Group — 策略训练组件
    # =========================================================================
    # 为什么需要这一步？
    # Actor 是整个系统的"大脑"——负责策略网络的参数更新。它：
    #   - 持有策略模型（如 ACT、OpenVLA、π0 等）的完整副本
    #   - 接收 Rollout 传来的 (obs, action, reward, advantage) 轨迹数据
    #   - 执行前向+反向传播，计算梯度并更新参数
    #   - 利用 FSDP 将模型分片到多 GPU，支持大模型训练
    #   - 定期将更新后的权重同步给 Rollout Worker
    #
    # "Group" 模式：create_group(cfg) 会根据配置创建一个 WorkerGroup 工厂，
    #   然后 .launch(cluster, ...) 在集群的指定节点上启动 1~N 个 Actor 副本。
    #   对于 FSDP，通常启动的是一组协作进程而非独立副本。
    # =========================================================================

    # 获取 Actor 组件的放置策略（决定部署在哪些节点/GPU）
    actor_placement = component_placement.get_strategy("actor")

    # 是否使用训练流水线模式（将前向/反向/梯度同步拆分为流水线阶段）
    # 目前只有默认的 embodied（PPO/GRPO 类）actor 支持 pipeline 模式
    use_training_pipeline = bool(cfg.runner.get("use_training_pipeline", False))

    # ---- 根据 loss_type 选择对应的 Actor Worker 实现类 ----
    # RLinf 支持多种算法，每种算法有专属的 Actor Worker 实现：
    #
    #   embodied_sac  → EmbodiedSACFSDPPolicy     (SAC: Soft Actor-Critic)
    #   embodied_dagger → EmbodiedDAGGERFSDPPolicy (DAGGER: 模仿学习)
    #   embodied_nft  → EmbodiedNFTFSDPPolicy      (NFT: Neural Field Transformer)
    #   其他(默认)    → EmbodiedFSDPActor          (PPO/GRPO/REINFORCE 等)
    #                 → PipelineEmbodiedFSDPActor  (流水线版本)
    #
    # 每种 Worker 内部实现了对应算法的损失计算和更新逻辑。
    # =========================================================================
    if cfg.algorithm.loss_type == "embodied_sac":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_sac."
            )
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        actor_worker_cls = EmbodiedSACFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_dagger":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_dagger."
            )
        from rlinf.workers.actor.fsdp_dagger_policy_worker import (
            EmbodiedDAGGERFSDPPolicy,
        )

        actor_worker_cls = EmbodiedDAGGERFSDPPolicy
    elif cfg.algorithm.loss_type == "embodied_nft":
        if use_training_pipeline:
            raise ValueError(
                "runner.use_training_pipeline=True is not supported for embodied_nft."
            )
        from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

        actor_worker_cls = EmbodiedNFTFSDPPolicy
    else:
        # 默认分支：PPO / GRPO / REINFORCE++ 等基于 Actor-Critic 或纯 Actor 的算法
        if use_training_pipeline:
            # 流水线模式：将 FSDP 训练拆分为 micro-batch 流水线，计算与通信重叠
            from rlinf.workers.actor.fsdp_actor_worker_pipeline import (
                PipelineEmbodiedFSDPActor,
            )

            actor_worker_cls = PipelineEmbodiedFSDPActor
        else:
            # 标准模式：完整的 FSDP 训练流程（前向→反向→梯度同步→优化器更新）
            from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

            actor_worker_cls = EmbodiedFSDPActor

    # create_group(cfg)：用配置构建 WorkerGroup（工厂模式）
    #   - 根据 cfg 确定需要创建多少个 Worker 实例
    #   - 为每个 Worker 准备初始化参数（模型路径、优化器配置、FSDP 参数等）
    # .launch(cluster, ...)：在 Ray 集群上真正启动这些 Worker
    #   - name="ActorGroup"：Worker 组的名称（用于日志和监控）
    #   - placement_strategy：决定这些 Worker 运行在哪些节点的哪些 GPU 上
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # =========================================================================
    # 步骤 4: 创建 Rollout Worker Group — 策略推演/数据收集组件
    # =========================================================================
    # 为什么需要这一步？
    # Rollout 的职责是用当前策略与环境交互，收集训练数据：
    #   a) 持有策略模型的推理副本（与 Actor 结构相同但权重独立）
    #   b) 从 Env 获取观测 obs，送入模型推理得到动作 action
    #   c) 将动作发送给 Env 执行，并收集 (obs, action, reward, done) 轨迹
    #   d) 计算 GAE 优势估计（如果配置了 GAE）
    #   e) 将处理好的轨迹批次发送给 Actor 进行训练
    #   f) 定期从 Actor 同步最新的模型权重
    #
    # MultiStepRolloutWorker：支持多步推演，可以在一个 rollout epoch 中
    #   连续执行多步交互，减少通信开销。
    # =========================================================================

    # 获取 Rollout 组件的放置策略
    rollout_placement = component_placement.get_strategy("rollout")

    # MultiStepRolloutWorker.create_group(cfg)：创建 Rollout Worker 组
    #   默认每个 GPU 创建一个 Rollout Worker 实例
    # .launch(...)：在集群上启动，使用 rollout_placement 指定的节点/GPU
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # =========================================================================
    # 步骤 5: 创建 Env Worker Group — 环境仿真组件
    # =========================================================================
    # 为什么需要这一步？
    # Env Worker 负责管理仿真环境实例：
    #   a) 持有环境实例（MuJoCo / IsaacSim / RoboTwin / ManiSkill2 等）
    #   b) 接收动作 action，执行 env.step(action)，返回 (next_obs, reward, done)
    #   c) 支持批量并行：一个 Env Worker 可管理多个环境实例（通过向量化）
    #   d) 处理环境 reset、终止检测、数据预处理
    #   e) 可选：录制环境视频、保存状态快照
    #
    # 为什么环境和推演要分离？
    #   这是 RLinf 的核心设计决策之一——"Disaggregated（分离式）"架构：
    #   - 仿真（Env）和推理（Rollout）解耦，可独立扩缩容
    #   - 允许 Env 运行在 CPU 节点，Rollout 运行在 GPU 节点
    #   - Env 可以预取下一批 obs，隐藏推理延迟
    #   - 适合异构集群：仿真节点不需要 GPU，推理节点必须有 GPU
    # =========================================================================

    # 获取 Env 组件的放置策略
    env_placement = component_placement.get_strategy("env")

    # EnvWorker.create_group(cfg)：根据 total_num_envs 和 group_size
    #   计算出需要多少个 Env Worker 实例
    # .launch(...)：在集群上启动这些 Worker
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    # =========================================================================
    # 可选: 创建 Reward Worker Group — 奖励模型组件
    # =========================================================================
    # 只有在显式启用 use_reward_model=True 且非独立真实世界模式时才创建。
    # Reward Worker 持有一个独立的奖励模型（如 learned reward function），
    # 用于替代或增强环境内置的稀疏奖励。
    # 典型场景：使用 VLM (Vision-Language Model) 作为奖励模型评估任务完成度。
    # =========================================================================
    reward_group = None
    if cfg.get("reward", {}).get("use_reward_model", False) and not cfg.get(
        "reward", {}
    ).get("standalone_realworld", False):
        # 获取 Reward 组件的放置策略
        reward_placement = component_placement.get_strategy("reward")
        reward_group = EmbodiedRewardWorker.create_group(cfg).launch(
            cluster, name=cfg.reward.group_name, placement_strategy=reward_placement
        )

    # =========================================================================
    # 步骤 6: 创建 EmbodiedRunner — 训练/评估主控制器
    # =========================================================================
    # 为什么需要这一步？
    # Runner 是整个训练流程的"指挥家"，它：
    #   a) 持有所有 Worker Group 的引用（Actor、Rollout、Env、Reward）
    #   b) 在内部创建 Channel（分布式数据通道），连接各组件
    #      - env_channel：Env → Rollout 的观测数据通道
    #      - rollout_channel：Rollout → Actor 的轨迹数据通道
    #      - actor_channel：Actor → Rollout 的权重更新/控制信号通道
    #      - reward_channel：Reward → Actor/Rollout 的奖励数据通道
    #   c) 实现训练/评估主循环逻辑（请看 EmbodiedRunner.run() 的实现）
    #   d) 管理 MetricLogger：记录训练指标到 TensorBoard/W&B/SwanLab
    #   e) 处理 checkpoint 的保存和恢复
    #   f) 协调权重同步节奏（weight_sync_interval）
    #
    # 这种设计将"流程编排"与"计算执行"分离：
    #   Runner = "做什么 + 什么顺序"（流程控制）
    #   Workers = "怎么做"（具体计算）
    # =========================================================================
    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
    )

    # =========================================================================
    # 步骤 7: runner.init_workers() — 初始化所有 Worker
    # =========================================================================
    # 为什么需要这一步？
    # 在 Worker 启动后（步骤 3-5），它们只是 Ray Actor 进程，还没有：
    #   a) 加载模型权重到 GPU（Actor 和 Rollout）
    #   b) 初始化优化器状态（Actor）
    #   c) 创建环境实例（Env）
    #   d) 建立 Channel 连接（各 Worker 之间的数据通道）
    #   e) 同步初始权重（Actor → Rollout）
    #
    # init_workers() 通过 RPC 调用每个 Worker 的 initialize() 方法：
    #   1. 调用 actor.initialize()：加载模型、初始化 FSDP、创建优化器
    #   2. 调用 rollout.initialize()：加载模型权重、初始化推理引擎
    #   3. 调用 env.initialize()：创建环境实例、设置种子
    #   4. 建立 Channel 连接：各 Worker 通过 Channel 可以收发数据
    #   5. 将 Actor 的初始权重同步给 Rollout（权重同步的第一步）
    #
    # 为什么要分两步（创建 → 初始化）？
    #   - Ray Actor 的创建是非阻塞的（异步创建进程）
    #   - 初始化是阻塞的（等待模型加载完成、CUDA 上下文就绪）
    #   - 分两步可以在所有 Worker 都就绪后统一初始化，避免时序竞争
    # =========================================================================
    runner.init_workers()

    # =========================================================================
    # 步骤 8: runner.run() — 执行训练/评估主循环
    # =========================================================================
    # 为什么需要这一步？
    # run() 是整个系统的核心，它执行以下循环（简化版）：
    #
    #   for epoch in range(max_epochs):
    #       for step in range(max_steps):
    #           ┌─────────────────────────────────────────┐
    #           │  1. Rollout 阶段（数据收集）              │
    #           │     - Env.reset() → obs                 │
    #           │     - 循环：env_channel.send(obs)        │
    #           │       Rollout.infer(obs) → action        │
    #           │       Env.step(action) → next_obs,r,d    │
    #           │     - 收集完整轨迹并计算 advantage        │
    #           │     - rollout_channel.send(trajectory)   │
    #           ├─────────────────────────────────────────┤
    #           │  2. Actor 阶段（策略更新）                │
    #           │     - rollout_channel.recv() → batch     │
    #           │     - Actor.train_step(batch)            │
    #           │       - 前向传播计算 loss                 │
    #           │       - 反向传播计算梯度                   │
    #           │       - FSDP all-reduce 梯度同步          │
    #           │       - 优化器更新参数                     │
    #           ├─────────────────────────────────────────┤
    #           │  3. 权重同步                              │
    #           │     - 每 weight_sync_interval 步          │
    #           │     - actor_channel.send(weights)         │
    #           │     - Rollout.load_weights(weights)      │
    #           ├─────────────────────────────────────────┤
    #           │  4. 评估阶段（每 val_check_interval）     │
    #           │     - 切换到 eval 模式                    │
    #           │     - 运行评估 episode                    │
    #           │     - 记录评估指标到 TensorBoard           │
    #           ├─────────────────────────────────────────┤
    #           │  5. 日志与保存                            │
    #           │     - MetricLogger 记录 train/eval 指标   │
    #           │     - 按 save_interval 保存 checkpoint   │
    #           └─────────────────────────────────────────┘
    #
    # 对于纯评估模式（only_eval: True）：
    #   - 跳过 Actor 训练阶段
    #   - 只执行 Rollout + Env + 评估日志
    #   - 加载 ckpt_path 指定的权重文件
    # =========================================================================
    runner.run()


if __name__ == "__main__":
    main()
