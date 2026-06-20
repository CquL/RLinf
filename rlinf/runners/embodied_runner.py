# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

# ============================================================================
# EmbodiedRunner — 具身智能训练/评估的"主控制器"
# ============================================================================
# 本文件是实现 RL 训练循环的核心。EmbodiedRunner = "指挥家"：
# 它自己不算、不仿真、不推理——它只负责"协调各 Worker 按正确顺序做事"。
#
# ============================================================================
# 阅读指南（先看哪里，后看哪里）
# ============================================================================
# 如果你是第一次读这个文件，建议按以下顺序看：
#
#   第 1 步：看 __init__()         → 了解 Runner 持有哪些资源（Worker、Channel、Logger）
#   第 2 步：看 set_max_steps()    → 了解训练要跑多少步
#   第 3 步：看 run()              → 核心！训练主循环，每个 step 发生了什么
#   第 4 步：看 update_rollout_weights() → Actor 如何把更新后的权重同步给 Rollout
#   第 5 步：看 evaluate()         → 评估流程
#   第 6 步：看 init_workers()     → Worker 初始化顺序
#   第 7 步：看 _log_step_metrics()→ 指标如何收集和记录
#   剩下的辅助函数用到时再看即可。
#
# ============================================================================
# 核心数据流（每个训练步 step）
# ============================================================================
#
#   ┌──────────────────────────────────────────────────────────────┐
#   │ 每个 step 做的事（run() 方法）                                  │
#   │                                                              │
#   │  ① sync_weights:     Actor ──(权重)──▶ Rollout               │
#   │                      （每 weight_sync_interval 步做一次）      │
#   │                                                              │
#   │  ② generate_rollouts: Env ◀──(通道)──▶ Rollout               │
#   │      Env 发出 obs → Rollout 推理得 action → Env 执行并返回    │
#   │      reward 和 done，循环直到 episode 结束                     │
#   │                                                              │
#   │  ③ cal_adv_and_returns: Actor 收到轨迹，计算 GAE 优势和回报    │
#   │                                                              │
#   │  ④ actor_training:     Actor 用优势+回报更新策略网络           │
#   │                                                              │
#   │  ⑤ eval & checkpoint:  定期评估 + 保存模型                    │
#   └──────────────────────────────────────────────────────────────┘
#
# ============================================================================
# 关键概念速查
# ============================================================================
# Channel  : 分布式"管道"。跨进程传递数据（基于 Ray 的分布式对象）。
#           类比：多线程里的 queue.Queue，但可以跨机器。
# Handle   : WorkerGroupFuncResult。调用 Worker 方法后返回的"远程操作句柄"。
#           调用 handle.wait() 阻塞等待远程操作完成并获取结果。
#           类比：Python 的 Future / JS 的 Promise。
# Timer    : 计时器，记录某段代码执行耗时，用于性能分析。
# ScopedTimer : 上下文管理器版本的计时器（with self.timer("名字"): ...）
# ============================================================================

import logging
import os
import queue
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

# Channel: 分布式通信通道，Worker 之间通过 Channel 收发数据
from rlinf.scheduler import Channel
# Handle: 远程调用句柄，调用 Worker 方法后立即返回 Handle，后续 .wait() 获取结果
from rlinf.scheduler import WorkerGroupFuncResult as Handle
# ScopedTimer: 上下文管理器，记录代码块的执行耗时
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
# MetricLogger: 指标记录器，将标量、时间等数据写入 TensorBoard / W&B / SwanLab
from rlinf.utils.metric_logger import MetricLogger
# compute_evaluate_metrics: 聚合多个评估环境的指标（如平均成功率）
# print_metrics_table: 在终端打印格式化的指标表格
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
# check_progress: 判断当前步是否需要做评估和保存 checkpoint
from rlinf.utils.runner_utils import check_progress
# Timer: 简单的计时器封装
from rlinf.utils.timers import Timer

logger = logging.getLogger(__name__)

# TYPE_CHECKING 块只在 IDE/类型检查器 中激活，运行时跳过。
# 这样既避免了循环导入，又让 IDE 能提供代码补全。
if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    # =========================================================================
    # __init__ — 初始化 Runner，组装所有资源
    # =========================================================================
    # 这个函数把所有"零部件"组装在一起，但还不开始训练。
    # 它做了什么：
    #   1. 保存所有 Worker 组件的引用（actor / rollout / env / reward）
    #   2. 创建 Channel（数据通道），连接各 Worker
    #   3. 创建 MetricLogger（指标记录器）和 Timer（计时器）
    #   4. 启动异步日志线程
    #   5. 计算 max_steps（总共要跑多少步）
    # =========================================================================
    def __init__(
        self,
        cfg: DictConfig,                                              # 完整的配置对象（Hydra 解析后的 OmegaConf DictConfig）
        actor: Union[                                                 # Actor Worker 组——负责训练策略网络
            "EmbodiedFSDPActor",
            "EmbodiedNFTFSDPPolicy",
            "EmbodiedSACFSDPPolicy",
            "AsyncEmbodiedSACFSDPPolicy",
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"], # Rollout Worker 组——负责推理和数据收集
        env: Union["EnvWorker", "AsyncEnvWorker"],                    # Env Worker 组——负责运行仿真环境
        reward: Union["EmbodiedRewardWorker"] = None,                 # Reward Worker 组（可选）——独立的奖励模型
        critic=None,                                                  # Critic Worker（保留参数，当前未使用）
    ):
        # ---- 保存所有组件的引用 ----
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward

        # 权重同步间隔：每隔多少步把 Actor 的最新权重推送给 Rollout
        # 如果值是 1，则每一步都同步（最安全但开销大）
        # 如果值是 N，则每 N 步同步一次（节省通信，但 Rollout 可能用"旧策略"采集数据）
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval

        # 是否允许环境 bootstrap（预取下一轮数据）与 Actor 训练重叠执行
        # 这是性能优化：Actor 训练时 Env 提前准备下一轮需要的初始状态
        self.overlap_env_bootstrap = bool(
            self.cfg.runner.get("overlap_env_bootstrap", False)
        )

        # ---- 性能分析 (profiling) 配置 ----
        # 用于特定 step 的详细性能分析（GPU 使用率、通信耗时等）
        # 从配置中读取 profiling 信息：哪些 step 需要 profile
        profiling_raw = self.cfg.cluster.get("profiling", None)
        profiling_enabled = profiling_raw is not None and bool(
            profiling_raw.get("enabled", True)                       # profiling 是否启用
        )
        profile_steps_raw = (
            profiling_raw.get("steps", None) if profiling_enabled else None  # 指定要 profile 的步号列表
        )
        # 如果 profiling 启用但没有指定 steps，则所有步都 profile
        self._profile_all_steps = profiling_enabled and profile_steps_raw is None
        # 如果指定了 steps，则转为 set[int] 用于快速查找
        self._profile_steps: set[int] | None = (
            {int(s) for s in profile_steps_raw}
            if profile_steps_raw is not None
            else None
        )

        # ---- 创建数据通道 (Channel) ----
        # Channel 是 Worker 之间的"数据管道"，基于 Ray 的分布式对象存储实现。
        # 每个 Channel 有一个名字（用于调试），支持多生产者-多消费者模式。
        #
        # 通道拓扑图：
        #
        #   env_channel:    Env ◀══════▶ Rollout
        #                     (obs, reward, done)       (actions)
        #
        #   rollout_channel: Rollout ◀════▶ Actor
        #                       (轨迹数据 batch)   (训练指令/权重)
        #
        #   actor_channel:   Actor ◀══════▶ Rollout
        #                     (权重同步信号)
        #
        #   reward_channel:  Reward ◀════▶ Env
        #                     (奖励计算请求/结果)
        #
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.reward is not None:
            self.reward_channel = Channel.create("Reward")
        else:
            self.reward_channel = None

        # 运行计时器，用于追踪整体训练是否超时
        self.run_timer = Timer(None)

        # consumed_samples: 已消费的训练样本总数（用于追踪训练进度）
        self.consumed_samples = 0
        # global_step: 全局训练步数计数器
        # 注意注释说"这里的 step 是 GRPO step"——这是因为 GRPO 算法的 step 定义略有不同
        self.global_step = 0

        # 根据配置计算 max_steps（训练总步数）
        self.set_max_steps()

        # ScopedTimer: 支持上下文管理器语法的计时器
        # reduction="max": 多次计时的同名操作取最大值（也可以取 mean/sum）
        # sync_cuda=False: 不强制 CUDA 同步（避免计时本身拖慢训练）
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        # ---- 日志系统 ----
        self.logger = get_logger()
        # MetricLogger: 将指标写入 TensorBoard / W&B / SwanLab 等后端
        self.metric_logger = MetricLogger(cfg)
        # 是否启用每个 Worker 独立记录指标（per_worker_log）
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        # ---- 异步日志线程 ----
        # 日志打印（尤其是格式化表格）比较慢，放到后台线程执行，
        # 避免阻塞训练主循环。
        self.stop_logging = False                                    # 控制日志线程退出的标志
        self.log_queue = queue.Queue()                               # 线程安全队列，存放待执行的日志任务
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)  # daemon=True 保证主进程退出时自动结束
        self.log_thread.start()

    # =========================================================================
    # _log_worker — 后台日志线程
    # =========================================================================
    # 这是一个在后台运行的线程，不断从 log_queue 中取出日志任务并执行。
    # 为什么要用线程？因为 print_metrics_table 可能涉及 I/O 操作（写终端/文件），
    # 放到后台做不会阻塞训练主循环。
    # =========================================================================
    def _log_worker(self):
        """后台线程：处理异步日志消息队列。"""
        while not self.stop_logging:                                 # 只要没被通知停止，就一直循环
            try:
                # 从队列中取日志任务，timeout=0.1 秒
                # 如果 0.1 秒内没有新任务，抛出 queue.Empty 异常
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)                                      # 执行日志函数（例如 print_metrics_table）
                self.log_queue.task_done()                           # 标记任务完成
            except queue.Empty:                                      # 队列空，继续等待
                continue
            except Exception as e:                                   # 日志出错不影响训练
                print(f"Logging error: {e}")
                continue

    # =========================================================================
    # print_metrics_table_async — 异步打印指标表格
    # =========================================================================
    # 将打印任务放入队列，由后台线程执行。避免阻塞训练循环。
    # =========================================================================
    def print_metrics_table_async(
        self,
        step: int,                                                   # 当前步数
        total_steps: int,                                            # 总步数
        start_time: float,                                           # 训练开始时间（用于计算 ETA）
        metrics: dict,                                               # 要显示的指标字典
        start_step: int = 0,                                         # 起始步数
    ):
        """将指标表格打印放入异步队列。"""
        self.log_queue.put(
            (
                print_metrics_table,                                 # 要调用的函数
                (                                                    # 函数的参数元组
                    step,
                    total_steps,
                    start_time,
                    metrics,
                    start_step,
                    self.metric_logger.log_path,
                ),
            )
        )

    # =========================================================================
    # init_workers — 初始化所有 Worker
    # =========================================================================
    # 这是在所有 Worker 进程创建完毕（通过 Ray 启动）后调用的第一步。
    # 初始化做了三件事：
    #   1. 让各 Worker 加载模型/环境等重资源
    #   2. 建立 Worker 之间的 Channel 通信链路
    #   3. 如果 resume_dir 指定了，恢复 checkpoint
    #
    # 初始化顺序很重要：
    #   - Rollout 和 Env 先初始化（可以并行）
    #   - Actor 最后初始化（因为 Actor 可能要等 Rollout 就绪后才能同步初始权重）
    #
    # Handle.wait() = 阻塞等待远程操作完成。
    #   例如 rollout_handle.wait() 会阻塞当前线程，直到 Rollout Worker 的
    #   init_worker() 方法在远程进程执行完毕。
    # =========================================================================
    def init_workers(self):
        # ---- 阶段 1：启动初始化（异步） ----
        # 调用 Rollout Worker 的 init_worker() 方法，立即返回 Handle
        rollout_handle = self.rollout.init_worker()
        # 调用 Env Worker 的 init_worker()，与 Rollout 并行初始化
        env_handle = self.env.init_worker()
        # 如果配置了独立的 Reward Worker，也一并初始化
        if self.reward is not None:
            self.reward.init_worker().wait()                         # wait() 阻塞直到完成

        # ---- 阶段 2：等待 Rollout 和 Env 初始化完成 ----
        rollout_handle.wait()
        env_handle.wait()

        # ---- 阶段 3：初始化 Actor（最后） ----
        # Actor 最后初始化是因为它可能需要加载 checkpoint，
        # 而 checkpoint 的恢复依赖其他组件已就绪
        self.actor.init_worker().wait()

        # ---- 阶段 4：恢复 checkpoint（如果指定了 resume_dir） ----
        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:                                       # 没有 resume_dir = 从头训练
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        # 通知 Actor 加载 checkpoint 权重
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        # 从路径名中恢复 global_step
        # 路径格式: .../checkpoints/global_step_1000/
        # 提取 "1000" 作为当前的 global_step
        self.global_step = int(resume_dir.split("global_step_")[-1])

    # =========================================================================
    # update_rollout_weights — 将 Actor 的最新权重同步到 Rollout
    # =========================================================================
    # 这是策略训练中的关键步骤：
    #   - Actor 持有最新训练好的模型权重
    #   - Rollout 需要用最新策略来采集数据
    #   - 所以需要定期从 Actor 同步权重到 Rollout
    #
    # 调用链：
    #   rollout.sync_model_from_actor() → Rollout 准备好接收权重
    #   actor.sync_model_to_rollout()   → Actor 把权重发送出去
    #   两边都 wait() 完成后，Rollout 就和 Actor 使用相同的策略了
    # =========================================================================
    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()  # Rollout 端：准备接收
        actor_handle: Handle = self.actor.sync_model_to_rollout()     # Actor 端：发送权重
        actor_handle.wait()                                            # 等 Actor 发送完成
        rollout_handle.wait()                                          # 等 Rollout 接收完成

    # =========================================================================
    # evaluate — 执行一轮评估
    # =========================================================================
    # 评估流程（比训练简单——只需要 Env + Rollout，不需要 Actor 更新）：
    #
    #   1. 调用 env.evaluate()    → Env 进入评估模式，创建评估环境实例
    #   2. 调用 rollout.evaluate()→ Rollout 进入评估模式，用确定性推理（不采样）
    #   3. Env 通过 env_channel 发观测，Rollout 通过 rollout_channel 收观测、
    #      推理得动作、发回给 Env 执行。循环直到所有评估 episode 完成。
    #   4. 收集所有环境的结果，计算平均成功率等指标。
    #
    # 返回值是聚合后的评估指标 dict，例如：
    #   {"success_rate": 0.85, "avg_return": 123.4, "avg_episode_length": 200}
    # =========================================================================
    def evaluate(self):
        # Env 端：启动评估，建立通道连接
        # input_channel：Env 通过它把 (obs, reward, done) 发给 Rollout
        # rollout_channel：Env 通过它接收 Rollout 发来的 action
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        # Rollout 端：启动评估，建立通道连接
        # input_channel：Rollout 通过它接收 Env 发来的 obs
        # output_channel：Rollout 通过它把推理出的 action 发回给 Env
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        # 等待 Env 评估完成，获取所有环境的结果列表
        env_results = env_handle.wait()
        rollout_handle.wait()                                        # 等 Rollout 也完成
        # 过滤掉 None 结果（可能某些环境出错了）
        eval_metrics_list = [results for results in env_results if results is not None]
        # 聚合所有环境的评估指标（如平均成功率）
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    # =========================================================================
    # _log_ranked_metrics — 记录每个 Worker 实例（rank）的独立指标
    # =========================================================================
    # 当有多 GPU / 多 Worker 时，每个 Worker 实例有独立的 rank。
    # 这个函数把每个 rank 的指标分别记录到 MetricLogger，
    # 方便在 TensorBoard 中按 rank 对比（例如看不同 GPU 上的负载是否均衡）。
    #
    # 参数：
    #   metrics_list:    每个 rank 的指标字典列表 [rank0_metrics, rank1_metrics, ...]
    #   step:            当前全局步数
    #   prefix:          指标名前缀（如 "time/actor"）
    #   worker_group_name: Worker 组名（如 "ActorGroup"）
    # =========================================================================
    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        # 如果没启用 per_worker 日志，或没有数据，直接返回
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        # 遍历每个 rank 的指标
        for rank, metrics in enumerate(metrics_list):
            if not metrics:                                          # 跳过空的
                continue
            # 给每个指标名加前缀，例如 "time/actor/forward_ms"
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            # 写入 MetricLogger（最终到 TensorBoard）
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    # =========================================================================
    # _aggregate_numeric_metrics — 聚合多个 Worker 的数值指标（取平均）
    # =========================================================================
    # 输入：[{"loss": 1.0, "lr": 1e-4}, {"loss": 2.0, "lr": 1e-4}, ...]
    # 输出：{"loss": 1.5, "lr": 1e-4}
    #
    # 对每个 key，把所有 rank 的值取平均。
    # =========================================================================
    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)                           # key → [val1, val2, ...]
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)                    # 收集所有 rank 的值
        # 对每个 key 取平均
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    # =========================================================================
    # _process_ranked_numeric_results — 处理带 rank 信息的数值结果
    # =========================================================================
    # 这是上面两个函数的组合使用：
    #   1. 从结果列表中提取出每个 rank 的数值指标
    #   2. 聚合所有 rank 的平均值（供主日志使用）
    #   3. 保留每个 rank 的独立指标（供 per_worker 日志使用）
    #
    # 参数：
    #   results: 每个 Worker 返回的结果列表
    #     [{rank: 0, "some_field": {"loss": 1.0}}, ...]
    #   metric_field: 要提取的字段名（如 "training_metrics"）
    #
    # 返回：
    #   aggregated_metrics: 所有 rank 的平均值 dict
    #   ranked_metrics_list: 按 rank 索引的独立指标列表
    # =========================================================================
    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)                 # 提取指定字段
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)                         # 读取 rank 编号
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)          # 按 rank 分组

        # 聚合所有 rank 的平均值
        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        # 为每个 rank 分别聚合
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)] # 创建 rank 数量大小的列表
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    # =========================================================================
    # _process_ranked_eval_results — 处理带 rank 信息的评估结果
    # =========================================================================
    # 和 _process_ranked_numeric_results 几乎一样，唯一的区别：
    # 聚合时用 compute_evaluate_metrics()（评估专用聚合，如计算成功率）
    # 而不是简单的数值平均。
    # =========================================================================
    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        # 用评估专用的聚合函数（如对成功率取平均）
        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    # =========================================================================
    # _split_pipeline_actor_results — 将流水线 Actor 的结果拆分为 rollout 和 training 两部分
    # =========================================================================
    # 流水线模式下，Actor 的 run_training() 返回的结果同时包含：
    #   - rollout_metrics:  数据收集阶段的指标（如轨迹长度、平均回报）
    #   - training_metrics: 训练阶段的指标（如 loss、grad_norm）
    # 这个函数把它们拆开，方便分别记录。
    # =========================================================================
    @staticmethod
    def _split_pipeline_actor_results(
        results: list[dict] | None,
    ) -> tuple[list[dict], list[dict]]:
        if not results:
            return [], []
        rollout_metrics = [result.get("rollout_metrics", {}) for result in results]
        training_metrics = [result.get("training_metrics", {}) for result in results]
        return rollout_metrics, training_metrics

    # =========================================================================
    # _maybe_eval_and_checkpoint — 在适当的时候执行评估和保存 checkpoint
    # =========================================================================
    # 由 check_progress() 判断当前步是否需要：
    #   - run_val:   运行一次评估（每 val_check_interval 步）
    #   - save_model: 保存模型 checkpoint（每 save_interval 步）
    #
    # 评估前会先同步权重（update_rollout_weights），
    # 确保评估用的是最新策略。
    # =========================================================================
    def _maybe_eval_and_checkpoint(self, step: int) -> dict:
        # check_progress 返回 (是否评估, 是否保存, 是否记录)
        run_val, save_model, _ = check_progress(
            self.global_step,
            self.max_steps,
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
            1.0,
            run_time_exceeded=False,
        )

        eval_metrics = {}
        if run_val:                                                  # 如果这步需要评估
            with self.timer("eval"):                                 # 计时
                self.update_rollout_weights()                        # 先同步权重
                eval_metrics = self.evaluate()                       # 执行评估
                # 给指标加 eval/ 前缀，如 "eval/success_rate"
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                self.metric_logger.log(data=eval_metrics, step=step) # 记录到 TensorBoard

        if save_model:                                               # 如果这步需要保存
            self._save_checkpoint()

        return eval_metrics

    # =========================================================================
    # _log_step_metrics — 聚合并记录当前步的所有指标（时间、环境、训练、评估）
    # =========================================================================
    # 这个函数做的事：
    #   1. 收集各 Worker 的耗时指标（env 耗时、rollout 耗时、actor 耗时）
    #   2. 收集各 Worker 的环境/训练/rollout 指标
    #   3. 加前缀分类（time/、env/、rollout/、train/、eval/）
    #   4. 写入 MetricLogger → TensorBoard
    #   5. 如果开启了 per_worker_log，也按 rank 分别记录
    #   6. 打印终端指标表格
    # =========================================================================
    def _log_step_metrics(
        self,
        step: int,                                                   # 当前训练步
        start_time: float,                                           # 训练开始时间
        start_step: int,                                             # 起始步
        env_handle: Handle,                                          # 当前步的 Env 操作句柄
        rollout_handle: Handle,                                      # 当前步的 Rollout 操作句柄
        actor_training_handle: Handle,                               # 当前步的 Actor 训练操作句柄
        reward_handle: Handle | None,                                # 当前步的 Reward 操作句柄
        actor_rollout_metrics: list[dict],                           # Actor 返回的 rollout 相关指标
        actor_training_metrics: list[dict],                          # Actor 返回的训练相关指标
        eval_metrics: dict,                                          # 评估指标
    ) -> None:
        # ---- 第 1 部分：收集时间指标 ----
        # self.timer 记录的 Runner 端耗时（如 sync_weights、generate_rollouts）
        time_metrics = self.timer.consume_durations()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

        # 从各 Worker 的 Handle 中提取 Worker 端耗时
        env_time_metrics, env_time_metrics_per_rank = env_handle.consume_durations(
            return_per_rank=True
        )
        rollout_time_metrics, rollout_time_metrics_per_rank = (
            rollout_handle.consume_durations(return_per_rank=True)
        )
        actor_time_metrics, actor_time_metrics_per_rank = (
            actor_training_handle.consume_durations(return_per_rank=True)
        )

        # 合并时间指标（都加上前缀）
        time_metrics.update({f"time/env/{k}": v for k, v in env_time_metrics.items()})
        time_metrics.update(
            {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
        )
        time_metrics.update(
            {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
        )
        if self.reward is not None:
            assert reward_handle is not None
            reward_time_metrics, reward_time_metrics_per_rank = (
                reward_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/reward/{k}": v for k, v in reward_time_metrics.items()}
            )

        # ---- 第 2 部分：收集环境指标 ----
        # env_results 是 Env Worker 返回的每个环境的统计信息
        # 如 success_rate、avg_return、episode_length 等
        env_results = env_handle.wait()
        env_results_list = [results for results in env_results if results is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)
        env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}

        # 为 per_worker 日志准备按 rank 分组的环境指标
        ranked_env_results = [
            {"rank": rank, "env": rank_metrics}
            for rank, rank_metrics in enumerate(env_results)
            if rank_metrics is not None
        ]
        _, env_metrics_per_rank = self._process_ranked_eval_results(
            ranked_env_results, metric_field="env"
        )

        # ---- 第 3 部分：收集 Rollout 和训练指标 ----
        rollout_metrics = {
            f"rollout/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_rollout_metrics).items()
        }
        training_metrics = {
            f"train/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_training_metrics).items()
        }

        # ---- 第 4 部分：写入所有指标到 MetricLogger ----
        self.metric_logger.log(env_metrics, step)
        self.metric_logger.log(rollout_metrics, step)
        self.metric_logger.log(time_metrics, step)
        self.metric_logger.log(training_metrics, step)

        # ---- 第 5 部分：按 rank 记录独立指标（per_worker_log） ----
        self._log_ranked_metrics(
            metrics_list=actor_rollout_metrics,
            step=step, prefix="rollout",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_training_metrics,
            step=step, prefix="train",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_time_metrics_per_rank,
            step=step, prefix="time/actor",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=rollout_time_metrics_per_rank,
            step=step, prefix="time/rollout",
            worker_group_name=self.rollout.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_time_metrics_per_rank,
            step=step, prefix="time/env",
            worker_group_name=self.env.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_metrics_per_rank,
            step=step, prefix="env",
            worker_group_name=self.env.worker_group_name,
        )
        if self.reward is not None:
            self._log_ranked_metrics(
                metrics_list=reward_time_metrics_per_rank,
                step=step, prefix="time/reward",
                worker_group_name=self.reward.worker_group_name,
            )

        # ---- 第 6 部分：合并所有指标用于终端表格打印 ----
        logging_metrics = time_metrics
        logging_metrics.update(eval_metrics)
        logging_metrics.update(env_metrics)
        logging_metrics.update(rollout_metrics)
        logging_metrics.update(training_metrics)

        # 异步打印终端表格（不阻塞主循环）
        self.print_metrics_table_async(
            step, self.max_steps, start_time, logging_metrics, start_step
        )

    # =========================================================================
    # _finish_run — 训练结束后的清理工作
    # =========================================================================
    def _finish_run(self) -> None:
        self.metric_logger.finish()                                   # 关闭 MetricLogger（flush 所有缓冲的指标）

        # 停止异步日志线程
        self.stop_logging = True                                     # 通知日志线程退出
        self.log_queue.join()                                        # 等待队列中所有任务被处理完
        self.log_thread.join(timeout=1.0)                            # 等待日志线程退出（最多 1 秒）

    # =========================================================================
    # _should_profile_step — 判断当前步是否需要性能分析
    # =========================================================================
    def _should_profile_step(self, step_idx: int) -> bool:
        return self._profile_all_steps or (
            self._profile_steps is not None and step_idx in self._profile_steps
        )

    # =========================================================================
    # _open_profiling_window — 打开性能分析窗口
    # =========================================================================
    # 通知所有 Worker 开始性能分析（如开始记录 CUDA kernel 耗时）。
    # 通常用 Nsight Systems 或 PyTorch Profiler 进行 GPU 级别的 profiling。
    # =========================================================================
    def _open_profiling_window(self, step_idx: int) -> None:
        self.logger.info(f"Opening profiling window at step {step_idx}")
        self.actor.start_profile(step_idx).wait()
        self.rollout.start_profile(step_idx).wait()
        self.env.start_profile(step_idx).wait()

    # =========================================================================
    # _close_profiling_window — 关闭性能分析窗口
    # =========================================================================
    def _close_profiling_window(self, step_idx: int) -> None:
        self.actor.stop_profile().wait()
        self.rollout.stop_profile().wait()
        self.env.stop_profile().wait()
        self.logger.info(f"Closed profiling window at step {step_idx}")

    # =========================================================================
    # =========================================================================
    # run() — 训练主循环（标准模式）  ★★★ 核心函数 ★★★
    # =========================================================================
    # =========================================================================
    # 这是整个文件最重要的函数。它实现了完整的 RL 训练循环。
    #
    # 伪代码概览：
    #   for step in range(max_steps):
    #       ① 同步权重（Actor → Rollout）
    #       ② 采集数据（Env ⟷ Rollout 交互，收集轨迹）
    #       ③ 计算优势（Actor 收到轨迹，算 GAE advantages）
    #       ④ 更新策略（Actor 做梯度下降）
    #       ⑤ 评估 & 保存 checkpoint
    #       ⑥ 记录指标到 TensorBoard
    #
    # 每个 step 的时间线（简化）：
    #
    #   |── sync ──|── generate_rollouts ──|── cal_adv ──|── training ──|── log ──|
    #   |          |  Env↔Rollout 多轮交互   |  GAE/GRPO   |  forward+back |         |
    #
    # 注意：如果配置了 use_training_pipeline=True，会跳到 run_pipeline()，
    # 那里的时序不同（rollout 和 training 可以重叠）。
    # =========================================================================
    def run(self):
        # 如果配置了流水线模式，走专门的处理路径
        if self.cfg.runner.get("use_training_pipeline", False):
            return self.run_pipeline()

        # =====================================================================
        # 标准模式训练循环
        # =====================================================================
        start_step = self.global_step                                  # 记录起始步（恢复训练时可能 > 0）
        start_time = time.time()                                       # 记录开始时间（用于计算 ETA）
        for _step in range(start_step, self.max_steps):               # 主训练循环
            # ---- 步骤 0：设置全局步数 ----
            # 每次循环开始时，告知 Actor 和 Rollout 当前的 global_step
            # 用途：某些算法（如学习率衰减）需要知道当前步数
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            # ---- 性能分析开关 ----
            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            # with self.timer("step") 包裹整个 step，记录单步总耗时
            with self.timer("step"):
                

                # =============================================================
                # 子步骤 ①：同步权重 (sync_weights)
                # =============================================================
                # 每隔 weight_sync_interval 步，把 Actor 的最新模型权重推送给 Rollout。
                # 这是为了保证 Rollout 采集数据时的策略和 Actor 训练的策略尽量一致。
                #
                # 为什么不是每一步都同步？
                #   - 同步需要跨网络传输大模型参数（可达数 GB），通信开销大
                #   - 对于 on-policy 算法（如 PPO），需要频密同步
                #   - 对于 GRPO 等允许一定 off-policy 的算法，可以降低同步频率
                # =============================================================
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()

                # =============================================================
                # 子步骤 ②：采集数据 (generate_rollouts)
                # =============================================================
                # 这是数据收集阶段。Env 和 Rollout 通过 Channel 进行多轮交互：
                #
                #   Env 端 (env.interact):
                #     while 没达到 max_steps_per_rollout:
                #         通过 env_channel 发 obs 给 Rollout
                #         通过 rollout_channel 收 action
                #         执行 env.step(action)
                #         收集 (obs, action, reward, done) 轨迹
                #
                #   Rollout 端 (rollout.generate):
                #     while 没收到停止信号:
                #         通过 rollout_channel 收 obs
                #         模型推理得到 action
                #         通过 env_channel 发 action 给 Env
                #         通过 actor_channel 把轨迹数据发给 Actor
                #
                # 注意这里的关键设计：
                #   - env.interact() 和 rollout.generate() 是并行启动的
                #   - 它们通过 Channel 进行乒乓式通信
                #   - 两边各自 wait() 等待完成后，整个 rollout 阶段结束
                # =============================================================
                with self.timer("generate_rollouts"):
                    # 启动 Env 的交互循环（非阻塞，立即返回 Handle）
                    env_handle: Handle = self.env.interact(
                        input_channel=self.env_channel,              # Env 通过它发送 obs
                        rollout_channel=self.rollout_channel,        # Env 通过它接收 action
                        reward_channel=self.reward_channel,          # 如果有独立 reward，通过它发送奖励请求
                        actor_channel=self.actor_channel,            # 轨迹数据发给 Actor
                    )
                    # 启动 Rollout 的生成循环（非阻塞，与 Env 并行运行）
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.rollout_channel,          # Rollout 通过它接收 obs
                        output_channel=self.env_channel,             # Rollout 通过它发送 action
                    )
                    # 如果配置了独立的 Reward Worker，启动奖励计算循环
                    reward_handle = None
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.reward_channel,
                            output_channel=self.env_channel,
                        )
                    # 通知 Actor 准备接收轨迹数据
                    # 等 Actor 确认准备好后，Rollout 才能开始发送
                    self.actor.recv_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()
                    # 等 Rollout 生成完成
                    rollout_handle.wait()
                    # 等 Reward Worker 完成（如果配置了）
                    if self.reward is not None:
                        reward_handle.wait()

                # =============================================================
                # 子步骤 ③：计算优势和回报 (cal_adv_and_returns)
                # =============================================================
                # Actor 收到轨迹数据后，计算 GAE（Generalized Advantage Estimation）
                # 或 GRPO 等算法的优势函数。
                #
                # 输入：轨迹中的 rewards, values（如果有 Critic）, dones, ...
                # 输出：advantages（每个时间步的优势值）和 returns（累积回报）
                #
                # advantages 告诉策略：当前动作比平均水平好多少（正值=好，负值=差）
                # returns 告诉策略：当前状态的总累积回报是多少
                # =============================================================
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # =============================================================
                # 子步骤 ④：更新策略 (actor_training)
                # =============================================================
                # Actor 用计算好的 advantages 和 returns 来更新策略网络：
                #   1. 前向传播：计算 policy loss（如 PPO loss、GRPO loss）
                #   2. 反向传播：计算梯度
                #   3. FSDP all-reduce：所有 GPU 同步梯度
                #   4. 优化器更新：Adam/SGD 更新模型参数
                #
                # 注意这里用到了 overlap_env_bootstrap 优化：
                #   - 如果启用，Actor 训练的同时 Env 可以提前准备下一轮数据
                #   - 这是一种流水线优化，让计算和 I/O 重叠
                # =============================================================
                actor_training_handle: Handle = self.actor.run_training()
                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    # 在 Actor 训练的同时，Env 预取下一轮需要的 bootstrap 数据
                    # 这样下一轮的 Env 初始化可以更快
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                # 等待 Actor 训练完成，获取训练指标（如 loss、grad_norm、lr）
                actor_training_metrics = actor_training_handle.wait()
                # 如果启动了 bootstrap 预取，也等它完成
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                # ---- 步数递增 ----
                self.global_step += 1

                # =============================================================
                # 子步骤 ⑤：评估 & 保存 checkpoint
                # =============================================================
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            # ---- 关闭性能分析窗口 ----
            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            # =============================================================
            # 子步骤 ⑥：记录所有指标
            # =============================================================
            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=reward_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        # 训练循环结束，清理
        self._finish_run()

    # =========================================================================
    # =========================================================================
    # run_pipeline() — 训练主循环（流水线模式）
    # =========================================================================
    # =========================================================================
    # 流水线模式和标准模式的核心区别：rollout 和 training 可以并行执行。
    #
    # 标准模式（串行）：
    #   |── rollout ──|── training ──|── rollout ──|── training ──|
    #
    # 流水线模式（并行）：
    #   |── rollout (step N) ──────|── rollout (step N+1) ────|
    #         |── training (step N-1) ──|── training (step N) ──|
    #
    # 这种模式下，Actor 的 run_training() 内部已经包含了：
    #   - 从 Channel 接收轨迹数据
    #   - 计算 advantages
    #   - 执行训练更新
    # 所以 Runner 不需要单独调用 compute_advantages_and_returns() 和 recv_rollout_trajectories()。
    #
    # 流水线模式在以下场景有优势：
    #   - Rollout 和 Training 在时间上重叠，提高 GPU 利用率
    #   - 适合异步 RL 算法
    #   - 但需要更复杂的同步逻辑
    # =========================================================================
    def run_pipeline(self):
        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            # 设置全局步数
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step"):
                # ---- 同步权重 ----
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()

                # ---- 启动 Env、Rollout（和标准模式相同） ----
                env_handle: Handle = self.env.interact(
                    input_channel=self.env_channel,
                    rollout_channel=self.rollout_channel,
                    reward_channel=self.reward_channel,
                    actor_channel=self.actor_channel,
                )
                rollout_handle: Handle = self.rollout.generate(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                )
                reward_handle = None
                if self.reward is not None:
                    reward_handle: Handle = self.reward.compute_rewards(
                        input_channel=self.reward_channel,
                        output_channel=self.env_channel,
                    )

                # ---- 启动 Actor 训练（流水线模式：不等 rollout 完成就开始） ----
                # 关键差异！Actor 的 run_training() 在流水线模式下会：
                #   1. 自己从 Channel 接收轨迹数据
                #   2. 自己计算 advantages
                #   3. 执行训练更新
                # 所有这些和 rollout 的数据收集重叠执行。
                actor_training_handle: Handle = self.actor.run_training(
                    input_channel=self.actor_channel                    # 从 Channel 自己取数据
                )

                # ---- 等待 Rollout 完成 ----
                with self.timer("generate_rollouts"):
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                # ---- Env bootstrap 预取（和标准模式相同） ----
                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                # ---- 等待 Actor 训练完成 ----
                # 此时 rollout 已经完成（数据已就绪），Actor 应该很快完成训练
                actor_results = actor_training_handle.wait()

                # 流水线模式下，Actor 返回的结果同时包含 rollout 和 training 两部分指标
                actor_rollout_metrics, actor_training_metrics = (
                    self._split_pipeline_actor_results(actor_results)
                )
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=reward_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()

    # =========================================================================
    # _save_checkpoint — 保存模型 checkpoint
    # =========================================================================
    # Checkpoint 保存路径结构：
    #   {log_path}/{experiment_name}/checkpoints/global_step_{N}/
    #   └── actor/   ← Actor 权重文件
    #
    # 之后可以用 resume_dir 指定这个路径来恢复训练。
    # =========================================================================
    def _save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        # 构建保存路径
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,                         # 如 "../results"
            self.cfg.runner.logger.experiment_name,                  # 如 "robotwin_beat_block_hammer_act_eval"
            f"checkpoints/global_step_{self.global_step}",           # 如 "checkpoints/global_step_1000"
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)                  # 创建目录
        # 通知 Actor Worker 保存权重和优化器状态
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    # =========================================================================
    # set_max_steps — 根据配置计算训练总步数
    # =========================================================================
    # RLinf 有两种方式指定训练长度：
    #   1. max_epochs: 训练 epoch 数 → max_steps = num_steps_per_epoch × max_epochs
    #   2. max_steps:   直接指定训练步数 → 取两者中较小值
    #
    # num_steps_per_epoch 默认为 1，意味着 1 epoch = 1 step。
    # =========================================================================
    def set_max_steps(self):
        self.num_steps_per_epoch = 1                                 # 每个 epoch 的步数
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs  # 从 epoch 计算

        # 如果同时也设置了 max_steps（且 ≥0），取最小值
        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    # =========================================================================
    # epoch — 属性：当前 epoch 编号
    # =========================================================================
    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
