# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

# ============================================================================
# MultiStepRolloutWorker — Rollout（推理/数据收集）Worker
# ============================================================================
# 这个 Worker 是"推理工具人"——它不做训练，只做推理。
#
# 核心循环（每一步）：
#   ① 从 Channel 收 Env 发来的 obs（观测）
#   ② 调用策略模型推理 → action
#   ③ 把推理结果发回给 Env（通过 Channel）
#
# 和之前看的 ACTRLPolicy 的关系：
#   ACTRLPolicy.predict_action_batch() ← 就是这个 Worker 调用的！
#   这个 Worker 持有 self.hf_model = ACTRLPolicy 实例
#
# 阅读顺序：
#   1. __init__()           → 初始化配置和参数
#   2. init_worker()        → 加载模型，准备推理
#   3. predict()            → 核心：观测→动作
#   4. generate_one_epoch() → 训练时的推理循环
#   5. generate()           → generate_one_epoch 的外层包装
#   6. evaluate()           → 评估时的推理循环
#   7. sync_model_from_actor() → 从 Actor 同步最新权重
# ============================================================================

import copy
import gc
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
# RolloutResult: 推理结果的数据结构
#   包含: actions, prev_logprobs, prev_values, forward_inputs, versions
from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
# WeightSyncer: 负责从 Actor 接收权重更新
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
# get_model: 根据配置创建策略模型实例（返回 ACTRLPolicy 等）
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
# Channel: 跨进程水管
# Worker: Worker 基类
from rlinf.scheduler import Channel, Cluster, Worker
# CommMapper: 多 Worker 之间的通信映射（谁发给谁）
from rlinf.utils.comm_mapping import CommMapper
# HybridComponentPlacement: 组件放置策略
from rlinf.utils.placement import HybridComponentPlacement


# ============================================================================
# 辅助函数：评估时 ACT 用几个 action chunk
# ============================================================================
def _get_eval_num_action_chunks(cfg: DictConfig) -> int:
    # 如果 ACT 启用了 stepwise 模式，每步执行 1 个动作
    if cfg.actor.model.model_type == "act" and cfg.actor.model.get(
        "eval_stepwise", False
    ):
        return 1
    # 否则，用完整的 action chunk 数量（默认 50）
    return int(cfg.actor.model.num_action_chunks)


class MultiStepRolloutWorker(Worker):
    # =========================================================================
    # __init__ — 初始化配置，不加载模型
    # =========================================================================
    # 主要做：
    #   1. 读取配置参数（batch_size, pipeline_stage_num, ...）
    #   2. 计算训练/评估时的 chunk 步数
    #   3. 初始化权重同步器（WeightSyncer）
    # =========================================================================
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        # Actor Worker 的组名（用于权重同步时找到 Actor）
        self.actor_group_name = cfg.actor.group_name
        # 当前 Worker 使用的 GPU 设备
        self.device = self.torch_platform.current_device()

        # pipeline_stage_num: 流水线阶段数，通常为 1
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        # enable_offload: 是否在推理后将模型卸到 CPU（省显存）
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        # 组件放置策略（获取 Rollout Worker 的 world_size 等）
        self.placement = HybridComponentPlacement(cfg, Cluster())
        rollout_world_size = self.placement.get_world_size("rollout")

        # ---- 权重同步相关 ----
        # actor_weight_src_rank: 从 Actor 的哪个 rank 接收权重（通常是 rank 0）
        self.actor_weight_src_rank = 0
        # 所有 Rollout Worker 的 rank 列表
        self._weight_sync_rollout_ranks = list(range(rollout_world_size))
        # 只有 rank 0 的 Rollout Worker 负责发送确认（减少通信冗余）
        self._weight_sync_is_sender = self._rank == 0

        # rollout_epoch: 每次 generate() 调用跑几个 epoch（通常 1）
        self.rollout_epoch = cfg.algorithm.get("rollout_epoch", 1)
        # 是否收集 transitions（状态转移数据）
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        # 专家模型（DAGGER 模仿学习时用到，通常为 None）
        self.expert_model = None

        # ---- 环境数量 ----
        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs

        # ---- 每个 Rollout Worker 处理的 batch 大小 ----
        # 总环境数 / Rollout Worker 数量 / pipeline 阶段数
        # 例如: 32 envs / 4 rollout workers / 1 pipeline = 8 per worker
        self.train_batch_size = (
            self.total_num_train_envs // self._world_size // self.num_pipeline_stages
        )
        self.eval_batch_size = (
            self.total_num_eval_envs // self._world_size // self.num_pipeline_stages
        )
        # enable_cuda_graph: 是否用 CUDA Graph 加速推理
        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)
        # 是否需要评估模式
        self.enable_eval = cfg.runner.val_check_interval > 0 or cfg.runner.only_eval

        # ---- 每个 epoch 的 chunk 步数 ----
        # 训练时：max_steps_per_rollout_epoch / num_action_chunks
        # 例如: 400 步 / 50 chunk_size = 8 个 chunk
        # 意味着这个 worker 需要执行 8 次推理
        self.n_train_chunk_steps = (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        # 评估时的 action chunk 数量
        self.eval_num_action_chunks = _get_eval_num_action_chunks(cfg)
        # 评估时的 chunk 步数
        self.n_eval_chunk_steps = (
            cfg.env.eval.max_steps_per_rollout_epoch
            // self.eval_num_action_chunks
        )
        # 是否收集 prev_logprobs / prev_values（训练时需要，评估时不需要）
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        # version: 当前权重的版本号（每次同步权重后 +1）
        self.version = 0
        # 已完成的 episode 总数
        self.finished_episodes = None

        # ---- 权重同步器 ----
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer", default=None)
        assert weight_syncer_cfg is not None, (
            "rollout.weight_syncer config must be provided"
        )
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)
        self._sync_weight_comm_options = self.weight_syncer.comm_options

    # =========================================================================
    # init_worker — 加载模型，准备推理 ★ 初始化入口 ★
    # =========================================================================
    # 在 runner.init_workers() 时被调用。
    # 做的事：
    #   1. 用 actor 的模型配置创建策略模型实例（self.hf_model）
    #   2. 如果指定了 checkpoint，加载权重
    #   3. 如果指定了 expert_model（DAGGER），加载专家模型
    #   4. 设置为 eval 模式（推理不需要 dropout/batchnorm）
    #   5. 计算 env↔rollout 的通信映射（dst_ranks, src_ranks）
    # =========================================================================
    def init_worker(self):
        # ---- 创建策略模型 ----
        # 复制 Actor 的模型配置（保证和 Actor 的结构完全一致）
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            # 但精度和路径用自己的（rollout 可能用 fp16，actor 用 bf16）
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        # get_model: 根据配置创建模型实例
        # 对于 ACT：返回 ACTRLPolicy 对象
        self.hf_model: BasePolicy = get_model(rollout_model_config)

        # ---- 如果指定了 checkpoint，加载权重 ----
        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        # ---- 可选：加载专家模型（DAGGER 用） ----
        if self.cfg.rollout.get("expert_model", None):
            expert_model_config = copy.deepcopy(self.cfg.actor.model)
            with open_dict(expert_model_config):
                expert_model_config.precision = self.cfg.rollout.expert_model.precision
                expert_model_config.model_path = (
                    self.cfg.rollout.expert_model.model_path
                )
            self.expert_model = get_model(expert_model_config)
            if self.cfg.runner.get("expert_ckpt_path", None):
                expert_model_dict = torch.load(self.cfg.runner.expert_ckpt_path)
                self.expert_model.load_state_dict(expert_model_dict)

        # ---- 切换到 eval 模式 ----
        # 推理时不需要 dropout、batchnorm 训练模式
        self.hf_model.eval()
        if self.expert_model is not None:
            self.expert_model.eval()

        # ---- 可选：Torch 编译优化 ----
        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)

        # ---- 可选：CUDA Graph 加速（跳过 CPU 启动开销） ----
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

        # ---- 设置 env ↔ rollout 的通信映射 ----
        # dst_ranks: 这个 Rollout Worker 要把结果发给哪些 Env Worker
        # src_ranks: 这个 Rollout Worker 要从哪些 Env Worker 接收数据
        self.dst_ranks = {}
        self.src_ranks = {}
        if not self.cfg.runner.only_eval:
            self.dst_ranks["train"] = self._setup_dst_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            )
            self.src_ranks["train"] = self._setup_src_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            )
        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )

        # 设置采样参数（温度、top_k、top_p 等）
        self.setup_sample_params()

        # 如果配置了 offload，先把模型移到 CPU（省显存）
        if self.enable_offload:
            self.offload_model()

    # =========================================================================
    # setup_sample_params — 设置推理时的采样参数
    # =========================================================================
    # 训练和评估使用不同的采样参数：
    #   训练：可能用采样（do_sample=True），有温度
    #   评估：确定性推理（除非 temperature_eval > 0）
    #
    # 注意：对于 ACT 模型，这些参数大部分被忽略
    # （ACT 的 predict_action_batch 直接接受 mode="train"/"eval"）
    # =========================================================================
    def setup_sample_params(self):
        # 生成长度参数
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # 采样参数
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        # 训练采样参数
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }
        # 评估采样参数
        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        # DAGGER 采样参数（专家/学生混合策略的混合概率）
        if self.expert_model is not None:
            self._dagger_sampling_params = {
                "beta": self.cfg.algorithm.get("dagger", {}).get("init_beta", 0.5),
                "beta_schedule": "exponential",
                "beta_min": 0.05,
                "beta_decay": 0.99,
            }

    # =========================================================================
    # predict — 核心推理函数 ★ 最重要 ★
    # =========================================================================
    # 这个函数就是调用策略模型的 predict_action_batch() 方法。
    # 也就是你之前看过的 ACTRLPolicy.predict_action_batch()。
    #
    # 数据流：
    #   env_obs: {"states": (B,14), "main_images": (B,H,W,3), ...}
    #       │
    #       ▼
    #   self.hf_model.predict_action_batch(env_obs, mode=mode)
    #       │
    #       ▼
    #   actions: (B, 50, 14)  ← 动作
    #   result: {
    #     "prev_logprobs": (B, 700),
    #     "prev_values": (B, 1),
    #     "forward_inputs": {qpos, images, action},
    #     "expert_label_flag": False
    #   }
    #
    # 额外逻辑：如果有 expert_model（DAGGER），按概率 β 用专家代替学生
    # =========================================================================
    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # ---- 决定采样参数 ----
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        # 对于 ACT 等模型，直接把 mode 传进去
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.ACT,
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
            SupportedModel.DREAMZERO,
            SupportedModel.CNN_POLICY,
            SupportedModel.CFG_MODEL,
        ]:
            if self.cfg.algorithm.loss_type == "embodied_dagger":
                kwargs = {"mode": "eval"}      # DAGGER 用 eval 模式
            else:
                kwargs = {"mode": mode}        # ACT 等：直接传 "train" 或 "eval"

        # ---- DAGGER：以概率 β 使用专家模型 ----
        only_save_expert = self.cfg.algorithm.get("dagger", {}).get(
            "only_save_expert", True
        )
        if mode == "train" and self.expert_model is not None:
            use_expert = torch.rand(1).item() < self._dagger_sampling_params["beta"]
        else:
            use_expert = False

        # ---- 推理（不计算梯度） ----
        with torch.no_grad():
            expert_label_flag = False
            if use_expert:
                # 用专家模型推理
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs, **kwargs
                )
                expert_label_flag = True
            else:
                # ★ 这就是调用 ACTRLPolicy.predict_action_batch() 的地方！ ★
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs, **kwargs
                )

        # ---- 转换为 torch.Tensor（如果模型返回了 numpy array） ----
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = bool(expert_label_flag)
        return actions, result

    # =========================================================================
    # get_bootstrap_values — 获取最终状态的价值估计
    # =========================================================================
    # 在 episode 结束时，用最后一帧观测做一次推理，得到 value 或 0
    # 只有带 value_head 的模型才返回非零值。ACT 没有，所以返回 None
    # =========================================================================
    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        if final_obs is None:
            return None
        if not (hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head")):
            # ACT 没有 value head → 跳过
            return None
        with torch.no_grad():
            actions, result = self.predict(final_obs)
            if "prev_values" in result and result["prev_values"] is not None:
                final_values = result["prev_values"]
            else:
                final_values = torch.zeros_like(actions[:, :1], dtype=torch.float32)
        return final_values[:, :1].cpu().contiguous()

    # =========================================================================
    # sync_model_from_actor — 从 Actor 同步最新权重 ★
    # =========================================================================
    # 训练循环的第①步（sync_weights）就是这个。
    # 做的事：
    #   1. 从 Actor 接收最新的模型权重
    #   2. 应用到自己的 self.hf_model 上
    #   3. 更新版本号
    #   4. 清理 GPU 缓存
    # =========================================================================
    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""

        # ---- 接收函数：从 Actor 收权重 ----
        async def recv_func() -> Any:
            return await self.broadcast(
                None,
                groups=[
                    (self.actor_group_name, self.actor_weight_src_rank),  # 从 Actor rank 0
                    (self._group_name, self._weight_sync_rollout_ranks),  # 发给所有 Rollout
                ],
                src=(self.actor_group_name, self.actor_weight_src_rank),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        # ---- 发送确认函数 ----
        async def send_func(data: Any) -> None:
            if not self._weight_sync_is_sender:
                return
            actor_world_size = self.placement.get_world_size("actor")
            for actor_rank in range(actor_world_size):
                await self.send(
                    data,
                    dst_group_name=self.actor_group_name,
                    dst_rank=actor_rank,
                    async_op=True,
                    options=self._sync_weight_comm_options,
                ).async_wait()

        # ---- 初始化接收器（第一次同步时） ----
        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=self.hf_model.state_dict(),
                recv=recv_func,
                send=send_func,
            )

        # ---- 接收并应用权重 ----
        # apply: 收到权重 → 更新 self.hf_model 的参数
        applied_version = await self.weight_syncer.apply(self.hf_model, recv_func)
        self.version = applied_version  # 更新版本号
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        # 通知模型更新了 global_step（用于学习率调度等）
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(applied_version)

        gc.collect()
        self.torch_platform.empty_cache()

    # =========================================================================
    # generate_one_epoch — 训练时的单 epoch 推理循环 ★
    # =========================================================================
    # 这个函数就是 Rollout Worker 在训练时的"主循环"。
    #
    # 循环逻辑：
    #   外层 for: n_train_chunk_steps 次（例如 400 步 / 50 chunk = 8 次）
    #     内层 for: pipeline_stage_num 次（通常 1）
    #       ① 从 Channel 接收 Env 发来的 obs
    #       ② 调用 predict() 推理
    #       ③ 把推理结果打包成 RolloutResult
    #       ④ 通过 Channel 发回给 Env
    #
    # RolloutResult 包含：
    #   - actions: 动作本身
    #   - prev_logprobs: 旧策略给的 log 概率
    #   - prev_values: 旧策略给的 value（ACT 为 None）
    #   - bootstrap_values: 终态的价值估计（ACT 为 None）
    #   - forward_inputs: {qpos, images, action} 给 Actor 训练用
    #   - versions: 当前权重版本号
    # =========================================================================
    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        self.update_dagger_beta()  # DAGGER: 衰减专家混合概率（通常不执行）

        # ---- 主循环：收 obs → 推理 → 发结果 ----
        for _ in range(self.n_train_chunk_steps):
            for _ in range(self.num_pipeline_stages):
                # 第 1 步：从 Channel 接收 Env 发来的观测
                # input_channel = rollout_channel（水管 A：Env→Rollout）
                # env_output = {"obs": {...}, "final_obs": None}
                env_output = await self.recv_env_output(input_channel)

                # 第 2 步：推理 ★
                # 调用 self.predict() → self.hf_model.predict_action_batch()
                # actions: (B, 50, 14)  动作
                # result: {prev_logprobs, prev_values, forward_inputs}
                actions, result = self.predict(env_output["obs"])

                # 第 3 步：打包成 RolloutResult
                save_flags = None
                if result.get("expert_label_flag", False):
                    save_flags = torch.full(
                        (actions.shape[0], self.cfg.actor.model.num_action_chunks),
                        True, dtype=torch.bool, device=actions.device,
                    )
                rollout_result = RolloutResult(
                    actions=actions,                                   # 动作
                    prev_logprobs=result["prev_logprobs"]              # 旧策略 logprob
                    if self.collect_prev_infos else None,
                    prev_values=result["prev_values"]                  # 旧策略 value
                    if self.collect_prev_infos else None,
                    bootstrap_values=self.get_bootstrap_values(        # 终态 bootstrap
                        env_output.get("final_obs", None)
                    ),
                    save_flags=save_flags,
                    forward_inputs=result["forward_inputs"],           # 给 Actor 的输入
                    versions=torch.full_like(                          # 权重版本号
                        result["prev_logprobs"],
                        float(self.version),
                        dtype=torch.float32,
                    ),
                )

                # 第 4 步：通过 Channel 发回给 Env
                # output_channel = env_channel（水管 B：Rollout→Env）
                self.send_rollout_result(output_channel, rollout_result, mode="train")

        # ---- 末尾：额外的 pipeline stage 处理 ----
        for _ in range(self.num_pipeline_stages):
            env_output = await self.recv_env_output(input_channel)
            actions, result = self.predict(env_output["obs"])
            rollout_result = RolloutResult(
                actions=actions,
                prev_values=result["prev_values"] if self.collect_prev_infos else None,
                bootstrap_values=self.get_bootstrap_values(
                    env_output.get("final_obs", None)
                ),
            )
            self.send_rollout_result(output_channel, rollout_result, mode="train")

    # =========================================================================
    # generate — 训练时的外层包装 ★ Runner 调用的入口 ★
    # =========================================================================
    # Runner 的 run() 里调用的就是 self.rollout.generate(input_channel, output_channel)
    #
    # 做的事：
    #   1. 如果 enable_offload，先把模型从 CPU 移到 GPU
    #   2. 调用 generate_one_epoch() 跑 rollout_epoch 次
    #   3. 跑完后，如果 enable_offload，把模型移回 CPU 省显存
    # =========================================================================
    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,    # rollout_channel: 接收 Env 发来的 obs
        output_channel: Channel,   # env_channel: 把推理结果发回给 Env
    ):
        if self.enable_offload:
            self.reload_model()    # CPU → GPU

        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),   # 只有 rank 0 显示进度条
        ):
            await self.generate_one_epoch(input_channel, output_channel)

        if self.enable_offload:
            self.offload_model()   # GPU → CPU

    # =========================================================================
    # evaluate — 评估时的推理循环 ★
    # =========================================================================
    # 和 generate_one_epoch 的区别：
    #   1. 使用 mode="eval"（确定性推理，不加噪声）
    #   2. 不收集 prev_logprobs 和 forward_inputs（不需要训练）
    #   3. 只在 reset_eval_state() 重置缓存后才启动
    #
    # 循环逻辑：
    #   for _ in range(n_eval_chunk_steps):
    #     ① 收 obs
    #     ② 推理（eval 模式）
    #     ③ 发 action 给 Env
    # =========================================================================
    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            # 重置评估状态（stepwise 模式下的缓存）
            if hasattr(self.hf_model, "reset_eval_state"):
                self.hf_model.reset_eval_state()

            # 评估循环：收 obs → 推理 → 发 action
            for _ in range(self.n_eval_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    actions, _ = self.predict(env_output["obs"], mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    # =========================================================================
    # offload_model / reload_model — GPU ↔ CPU 模型移动
    # =========================================================================
    # enable_offload=True 时：推理前移到 GPU，推理后移回 CPU
    # 好处：省 GPU 显存（Rollout 不推理的时间段 GPU 可以给别人用）
    # 代价：移动需要时间
    # =========================================================================
    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

    # =========================================================================
    # recv_env_output — 从 Channel 接收 Env 发来的观测
    # =========================================================================
    # 处理多 Env Worker 场景：从一个或多个 Env rank 接收 obs，合并在一起
    # 输入：input_channel = rollout_channel（水管 A）
    # 输出：{"obs": {...}, "final_obs": None}
    # =========================================================================
    @Worker.timer("rollout/recv_obs")
    async def recv_env_output(
        self, input_channel: Channel, mode: Literal["train", "eval"] = "train"
    ) -> dict[str, Any]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        obs_batches = []
        for src_rank, expected_size in src_ranks_and_sizes:
            # 从 Channel 中获取（key 包含 rank 信息，防止拿错）
            obs_batch = await input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_obs"
                ),
                async_op=True,
            ).async_wait()
            actual_size = self._infer_env_batch_size(obs_batch)
            assert actual_size == expected_size
            obs_batches.append(obs_batch)
        # 合并多个 Env rank 的 obs
        return self._merge_obs_batches(obs_batches)

    # =========================================================================
    # send_chunk_actions — 把推理出的动作分片发给各 Env Worker
    # =========================================================================
    # 如果有多 Env Worker，需要把 (B, 50, 14) 按照 batch 维度拆开，
    # 分别发给对应的 Env Worker
    # =========================================================================
    @Worker.timer("rollout/send_actions")
    def send_chunk_actions(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | np.ndarray,
        mode: Literal["train", "eval"] = "train",
    ):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        # 按 batch 维度拆开
        chunk_actions_split = self._split_actions(chunk_actions, split_sizes)
        for (dst_rank, _), chunk_action_i in zip(
            dst_ranks_and_sizes, chunk_actions_split
        ):
            if isinstance(chunk_action_i, torch.Tensor):
                chunk_action_i = chunk_action_i.detach().cpu().contiguous()
            # 放到 Channel 里（key 含 rank 信息）
            output_channel.put(
                chunk_action_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_actions"
                ),
                async_op=True,
            )

    # =========================================================================
    # send_rollout_result — 把推理结果（RolloutResult）发给 Env
    # =========================================================================
    # 和 send_chunk_actions 的区别：
    #   这个是训练时用的，发整个 RolloutResult（含 logprob、forward_inputs）
    #   send_chunk_actions 是评估时用的，只发 actions
    # =========================================================================
    @Worker.timer("rollout/send_traj")
    def send_rollout_result(
        self,
        output_channel: Channel,
        rollout_result: RolloutResult,
        mode: Literal["train", "eval"] = "train",
    ):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        split_rollout_results = self._split_rollout_result(rollout_result, split_sizes)
        for (dst_rank, _), rollout_result_i in zip(
            dst_ranks_and_sizes, split_rollout_results
        ):
            output_channel.put(
                rollout_result_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_rollout_results"
                ),
                async_op=True,
            )

    # =========================================================================
    # 辅助函数：通信映射、数据拆分/合并
    # =========================================================================
    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """计算这个 Rollout Worker 要把结果发给哪些 Env Worker"""
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """计算这个 Rollout Worker 要从哪些 Env Worker 接收数据"""
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            dst_rank=self._rank,
        )

    def _split_actions(
        self, actions: torch.Tensor | np.ndarray, sizes: list[int]
    ) -> list[torch.Tensor | np.ndarray]:
        """把 (B_total, 50, 14) 按 batch 维度拆成多个小块"""
        assert sum(sizes) == actions.shape[0]
        if isinstance(actions, np.ndarray):
            split_indices = np.cumsum(sizes[:-1]).tolist()
            return list(np.split(actions, split_indices, axis=0))
        return list(torch.split(actions, sizes, dim=0))

    def _split_rollout_result(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        """把 RolloutResult 按 batch 维度拆成多个小块"""
        # 对每个字段按 batch 维度拆分
        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_save_flags = _split_optional_tensor(rollout_result.save_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                }
                for idx in range(len(sizes))
            ]
        )
        return [
            RolloutResult(
                actions=split_actions[idx],
                prev_logprobs=split_prev_logprobs[idx],
                prev_values=split_prev_values[idx],
                bootstrap_values=split_bootstrap_values[idx],
                save_flags=split_save_flags[idx],
                forward_inputs=split_forward_inputs[idx],
                versions=split_versions[idx],
            )
            for idx in range(len(sizes))
        ]

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        """从 obs 字典推断 batch 大小"""
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        """合并多个 Env Worker 发来的 obs（在 batch 维度拼接）"""
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)  # Tensor: cat 拼接
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]  # list: extend
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = [
                final_obs if final_obs is not None else obs_dict
                for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        return {"obs": merged_obs, "final_obs": merged_final_obs}

    def update_dagger_beta(self):
        """DAGGER: 指数衰减专家混合概率（训练步数越多，越少用专家）"""
        if self.expert_model is None:
            return
        if self._dagger_sampling_params["beta_schedule"] == "exponential":
            self._dagger_sampling_params["beta"] = max(
                self._dagger_sampling_params["beta_min"],
                self._dagger_sampling_params["beta"]
                * self._dagger_sampling_params["beta_decay"],
            )

    def set_global_step(self, global_step: int):
        """设置全局步数（传递给策略模型）"""
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
