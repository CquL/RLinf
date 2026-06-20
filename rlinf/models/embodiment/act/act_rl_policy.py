# ============================================================================
# ACTRLPolicy — 把 ACT（模仿学习模型）包装成可被 RL 训练的策略
# ============================================================================
# 阅读顺序建议：
#   1. __init__()                   → 看看它从 checkpoint 加载了什么
#   2. predict_action_batch()       → Rollout 推理时调用（核心！）
#   3. default_forward()            → Actor 训练时调用（算 logprob）
#   4. _action_distribution()       → 动作的"概率分布"是什么
#   5. _normalize_qpos / _denormalize_actions → 数据归一化
# ============================================================================
#
# 这个文件解决的核心问题：
# ─────────────────────────────
# ACT 原始模型是确定性模型（给定 obs，输出一个固定的 action）。
# 但强化学习需要策略能"探索"——即同一个 obs 下偶尔尝试不同的 action。
#
# 解决办法：在 ACT 的输出上加一个 Normal 分布。
#   ACT 的输出 → 作为分布的 mean（均值）
#   新加的参数 logstd → 作为分布的 std（标准差）
#   训练时从分布中采样（有随机性）    → exploration（探索）
#   评估时直接用 mean（确定性）       → exploitation（利用）
#
# 控制理论类比：
#   ACT 输出 = 标称控制量 u_nominal
#   logstd   = 控制量的噪声幅值
#   训练时：u = u_nominal + noise（探索新的控制策略）
#   评估时：u = u_nominal（执行已知最优策略）
# ============================================================================

"""RLinf adapter for the RoboTwin ACT policy checkpoint."""

import os
import pickle
import sys
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# BasePolicy: RLinf 的策略基类，定义了 predict_action_batch / default_forward 等接口
# ForwardType: 一个枚举，用来区分"训练前向"还是"推理前向"
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class ACTRLPolicy(nn.Module, BasePolicy):
    # =========================================================================
    # __init__ — 加载 ACT 模型 + 准备 RL 所需的额外参数
    # =========================================================================
    # 这个函数做的事（按顺序）：
    #   1. 读取配置（动作维度、相机名称、图像大小等）
    #   2. 加载原始 ACT 模型权重（从 RoboTwin 训练的 checkpoint）
    #   3. 加载数据统计量（action_mean/std, qpos_mean/std）用于归一化
    #   4. 创建 logstd（可学习参数）——这是给 RL 加的唯一新参数
    #   5. 准备图像归一化参数（ImageNet 的 mean/std）
    # =========================================================================
    def __init__(self, cfg, torch_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        # 数值精度：通常 bf16 或 fp32
        self.torch_dtype = torch_dtype or torch.float32

        # ---- 从配置读取关键参数 ----
        # action_dim: 每个时间步动作向量的维度（双臂 14 维 = 7 关节 × 2 臂）
        self.action_dim = int(cfg.get("action_dim", 14))
        # state_dim: 状态向量的维度（关节位置），通常和 action_dim 相同
        self.state_dim = int(cfg.get("state_dim", self.action_dim))
        # num_action_chunks: ACT 一次生成多少个连续动作块
        # 原始 ACT 是 50，意味着一次推理输出 50 个时间步的动作序列
        self.num_action_chunks = int(
            cfg.get("num_action_chunks", cfg.get("chunk_size", 50))
        )
        # camera_names: 使用哪些相机的图像作为输入
        # 典型值: ["cam_high", "cam_right_wrist", "cam_left_wrist"]
        #         ↑ 头部相机    ↑ 右腕相机         ↑ 左腕相机
        self.camera_names = list(
            cfg.get(
                "camera_names",
                ["cam_high", "cam_right_wrist", "cam_left_wrist"],
            )
        )

        # ---- 评估模式配置 ----
        # use_stochastic_eval: True = 评估时也采样（有随机性）
        #                      False = 评估时用 mean（确定性）
        self.use_stochastic_eval = bool(cfg.get("use_stochastic_eval", False))
        # eval_stepwise: True = 逐步执行动作（每步取 chunk 中的一个）
        #                False = 一次性输出整个 chunk 的动作
        self.eval_stepwise = bool(cfg.get("eval_stepwise", False))
        # temporal_agg: 时间聚合——用指数加权平均平滑连续多帧的预测
        # 减少动作抖动，类似于控制中的低通滤波器
        self.temporal_agg = bool(cfg.get("temporal_agg", False))
        self.temporal_agg_decay = float(cfg.get("temporal_agg_decay", 0.01))
        # 评估时的内部状态（用于 stepwise 推理和 temporal aggregation）
        self._eval_step = 0                                          # 当前评估到了第几步
        self._eval_action_cache: torch.Tensor | None = None          # 缓存的 action chunk
        self._eval_action_history: list[torch.Tensor] = []           # 历史预测（供 temporal agg 使用）

        # ---- 加载原始 ACT 模型 ----
        # ACT 的训练代码在 RoboTwin 仓库中，不在 RLinf 里
        # 所以需要 sys.path 导入外部代码
        act_source_dir = Path(os.path.expanduser(str(cfg.act_source_dir))).resolve()
        if not act_source_dir.exists():
            raise FileNotFoundError(f"ACT source dir does not exist: {act_source_dir}")
        sys.path.insert(0, str(act_source_dir))                     # 临时加入 Python 路径

        from detr.models import build_ACT_model                      # 来自 RoboTwin 的 ACT 构建函数

        act_args = self._build_act_args(cfg)                         # 构建模型配置参数
        self.act_model = build_ACT_model(act_args)                   # 创建 ACT 模型（CVAE + DETR-style decoder）
        self._load_checkpoint(str(cfg.model_path))                   # 加载预训练权重
        self.act_model.to(dtype=self.torch_dtype)                    # 转换精度

        # ---- 加载数据统计量（用于归一化和反归一化） ----
        # 为什么需要归一化？
        # 神经网络喜欢输入在 [-1, 1] 或 [0, 1] 范围。但真实的关节角度可能是 [-3.14, 3.14] 弧度。
        # 所以需要：原始值 → (x - mean) / std → 归一化值 → 输入网络
        #         网络输出归一化值 → x * std + mean → 原始值 → 发给机器人
        stats = self._load_stats(str(cfg.dataset_stats_path))

        # register_buffer: 把 tensor 注册为模型的一部分，但不是可学习参数
        # 好处：跟随模型一起 .to(device)、一起保存/加载
        #
        # action_mean/std shape: (1, 1, action_dim) = (1, 1, 14)
        # 为什么是 (1, 1, 14)？因为动作有 chunk_size 个时间步和 batch_size 个样本
        # 广播机制让它们可以和 (B, chunk_size, 14) 的 tensor 直接做运算
        self.register_buffer(
            "action_mean",
            self._stat_tensor(stats, "action_mean", [1, 1, self.action_dim]),
        )
        self.register_buffer(
            "action_std",
            self._stat_tensor(stats, "action_std", [1, 1, self.action_dim]),
        )
        # qpos_mean/std shape: (1, state_dim) = (1, 14)
        self.register_buffer(
            "qpos_mean",
            self._stat_tensor(stats, "qpos_mean", [1, self.state_dim]),
        )
        self.register_buffer(
            "qpos_std",
            self._stat_tensor(stats, "qpos_std", [1, self.state_dim]),
        )

        # ---- 创建 logstd：这是 RL 给 ACT 加的唯一可学习参数！ ----
        # 回顾一下：ACT 原始输出是确定性的 a_hat（归一化空间）。
        # RL 需要随机性来做探索。所以我们创建一个可学习的 logstd，
        # 把 a_hat 作为 Normal 分布的 mean，exp(logstd) 作为 std。
        #
        # 为什么学 log(std) 而不是直接学 std？
        #   std > 0 必须满足，直接学要加约束
        #   log(std) 可以取任意实数，然后 exp(logstd) 自然 > 0
        #
        # shape: (1, num_action_chunks, action_dim) = (1, 50, 14)
        # 每个 action chunk 的每一步、每个动作维度都有独立的 std
        # initial_logstd = -2.0 → 初始 std = exp(-2.0) ≈ 0.135（在归一化空间中）
        initial_logstd = float(cfg.get("initial_logstd", -2.0))
        self.logstd = nn.Parameter(                                    # nn.Parameter = 可学习的参数
            torch.full(
                (1, self.num_action_chunks, self.action_dim),          # shape (1, 50, 14)
                initial_logstd,                                        # 全部初始化为 -2.0
                dtype=torch.float32,
            )
        )
        # 限制 logstd 的取值范围，防止 std 过大（噪声太大）或过小（失去探索能力）
        self.min_logstd = float(cfg.get("min_logstd", -5.0))          # 最小 logstd，对应 std ≈ 0.0067
        self.max_logstd = float(cfg.get("max_logstd", 1.0))           # 最大 logstd，对应 std ≈ 2.72

        # ---- 图像归一化参数 ----
        # image_normalization: "imagenet" = 用 ImageNet 的 mean/std 归一化
        #                       "none"      = 不归一化
        self.image_normalization = str(cfg.get("image_normalization", "imagenet")).lower()
        if self.image_normalization not in {"none", "imagenet"}:
            raise ValueError(
                "ACT image_normalization must be one of {'none', 'imagenet'}, "
                f"got {self.image_normalization!r}."
            )
        # 图像缩放目标尺寸，例如 (480, 640) 表示高 480、宽 640
        self.image_resize_hw = cfg.get("image_resize_hw", None)
        if self.image_resize_hw is not None:
            self.image_resize_hw = tuple(int(size) for size in self.image_resize_hw)
            if len(self.image_resize_hw) != 2 or min(self.image_resize_hw) <= 0:
                raise ValueError(
                    "ACT image_resize_hw must be [height, width], "
                    f"got {self.image_resize_hw!r}."
                )

        # ImageNet 标准化的 mean 和 std（用于图像归一化）
        # shape: (1, 3, 1, 1) — 适配 (B, C, H, W) 图像格式
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    # =========================================================================
    # _build_act_args — 构建 ACT 模型的配置参数
    # =========================================================================
    # ACT 原始代码（RoboTwin）用 argparse.Namespace 传参数。
    # 这里把 RLinf 的 OmegaConf 配置转成 Namespace，保持兼容。
    # 大部分参数只在加载 checkpoint 时用到，RL 训练时不改变。
    # =========================================================================
    def _build_act_args(self, cfg) -> Namespace:
        """Create the Namespace expected by RoboTwin's ACT builder."""
        return Namespace(
            lr=float(cfg.get("lr", 1e-5)),
            lr_backbone=float(cfg.get("lr_backbone", 1e-5)),
            batch_size=int(cfg.get("batch_size", 1)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
            epochs=int(cfg.get("epochs", 1)),
            chunk_size=self.num_action_chunks,                         # 50: 每次输出 50 步动作
            kl_weight=float(cfg.get("kl_weight", 10.0)),               # CVAE 的 KL 散度权重
            hidden_dim=int(cfg.get("hidden_dim", 512)),                # Transformer 隐藏维度
            dim_feedforward=int(cfg.get("dim_feedforward", 3200)),     # FFN 中间维度
            backbone=str(cfg.get("backbone", "resnet18")),             # 视觉骨干网络
            enc_layers=int(cfg.get("enc_layers", 4)),                  # Transformer 编码器层数
            dec_layers=int(cfg.get("dec_layers", 7)),                  # Transformer 解码器层数
            nheads=int(cfg.get("nheads", 8)),                          # 多头注意力头数
            position_embedding=str(cfg.get("position_embedding", "sine")), # 位置编码方式
            dilation=bool(cfg.get("dilation", False)),                 # 卷积膨胀
            dropout=float(cfg.get("dropout", 0.1)),
            pre_norm=bool(cfg.get("pre_norm", False)),                 # Pre-LayerNorm
            masks=bool(cfg.get("masks", False)),                       # 注意力掩码
            camera_names=self.camera_names,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )

    # =========================================================================
    # _load_checkpoint — 加载 ACT 预训练权重
    # =========================================================================
    # 从 RoboTwin 训练的 .pth 文件加载权重。
    # 处理两种格式：
    #   1. {"model": {...}}  → 提取 "model" 键（常见于 torch.save({'model': ...})）
    #   2. {...}             → 直接使用（常见于 torch.save(model.state_dict())）
    # "strict=False" 允许部分键不匹配（RL 加了 logstd，原 checkpoint 里没有）
    # =========================================================================
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_path = os.path.expanduser(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"ACT checkpoint does not exist: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")   # 先加载到 CPU，避免 GPU 内存不够
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]                           # 提取嵌套的 model 字典
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported ACT checkpoint format: {type(state_dict)!r}")

        # 去掉可能存在的 "model." 前缀（有些 checkpoint 会用 model.encoder... 的键名）
        stripped_state = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                stripped_state[key.removeprefix("model.")] = value
            else:
                stripped_state[key] = value

        # strict=False: RL 新增的 logstd 不在原 checkpoint 中，允许缺失
        missing, unexpected = self.act_model.load_state_dict(stripped_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected ACT checkpoint keys: {unexpected[:10]}")
        if missing:
            raise RuntimeError(f"Missing ACT checkpoint keys: {missing[:10]}")

    # =========================================================================
    # _load_stats — 加载数据集统计量（归一化参数）
    # =========================================================================
    # 数据集统计量（dataset_stats.pkl）包含：
    #   action_mean, action_std: 动作空间上计算的均值和标准差
    #   qpos_mean, qpos_std:     关节空间上计算的均值和标准差
    # 这些值是在 ACT 训练数据集上预先统计好的。
    # =========================================================================
    def _load_stats(self, stats_path: str) -> dict[str, Any]:
        stats_path = os.path.expanduser(stats_path)
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"ACT dataset stats do not exist: {stats_path}")
        with open(stats_path, "rb") as stats_file:
            stats = pickle.load(stats_file)                            # .pkl 文件用 pickle 加载
        if not isinstance(stats, dict):
            raise TypeError(f"Unsupported ACT stats format: {type(stats)!r}")
        return stats

    # =========================================================================
    # _stat_tensor — 从 stats 字典中取出指定 key 并 reshape
    # =========================================================================
    def _stat_tensor(
        self, stats: dict[str, Any], key: str, target_shape: Sequence[int]
    ) -> torch.Tensor:
        if key not in stats:
            raise KeyError(f"ACT dataset stats missing key: {key}")
        value = torch.as_tensor(stats[key], dtype=torch.float32)
        return value.reshape(*target_shape)

    # =========================================================================
    # forward — 分发函数（RLinf 的统一入口）
    # =========================================================================
    # RLinf 的 Actor Worker 调用 forward(forward_type=ForwardType.DEFAULT, ...)
    # 这里把它路由到 default_forward()
    # =========================================================================
    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        """Dispatch RLinf forward calls."""
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"ACT does not support forward type {forward_type}.")

    # =========================================================================
    # =========================================================================
    # default_forward — 训练时调用：重新计算 logprob
    # =========================================================================
    # =========================================================================
    # 这个函数在 Actor Worker 训练阶段被调用。回顾训练循环：
    #   Rollout 阶段：策略采样了 action，记录了 forward_inputs 和 prev_logprobs
    #   Actor 阶段：用 forward_inputs 重新前向传播，计算当前策略的 logprob
    #
    # 为什么要"重新前向"？
    #   因为训练过程中策略参数变了。同一个 (obs, action) 组合，
    #   旧策略认为它的概率是 prev_logprob，
    #   新策略认为它的概率是 curr_logprob = default_forward 算出来的。
    #   ratio = exp(curr - prev) 表示"新策略比旧策略更/更不想做这个动作"。
    #
    # 数据流（带 shape）：
    #   输入 forward_inputs = {
    #     "qpos":  (B, state_dim)       归一化后的关节位置
    #     "images": (B, N_cam, 3, H, W) 相机图像
    #     "action": (B, chunk_size * action_dim) 当时采样的动作（已归一化，展平）
    #   }
    #
    #   处理流程：
    #     ① qpos + images → ACT 模型 → mean (B, chunk_size, action_dim)
    #     ② mean → Normal(mean, std=exp(logstd)) → 动作的概率分布
    #     ③ 算 log_prob(当时的 action 在这个分布下的对数概率)
    #     ④ 返回 {"logprobs": (B, chunk_size * action_dim)}
    # =========================================================================
    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,                                 # 是否计算 logprob（训练时需要，不需要时跳过省计算）
        compute_entropy: bool = False,                                 # 是否计算熵（衡量策略的随机程度）
        compute_values: bool = False,                                  # 是否计算 value（ACT 没有 Critic，总是 False）
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Recompute action log-probabilities for actor updates."""
        if compute_values:
            raise NotImplementedError("ACT adapter does not provide a value head.")
        if not compute_logprobs:                                       # 不需要算 logprob 时直接返回空
            return {}

        # ---- 从 forward_inputs 提取数据并移到 GPU ----
        # forward_inputs 在 Rollout 阶段保存在 CPU 上，这里 .to(device) 移到 GPU
        # qpos shape: (B, state_dim) = (1, 14)
        qpos = forward_inputs["qpos"].to(self.device, dtype=self.torch_dtype)
        # images shape: (B, N_cam, 3, H, W) = (1, 3, 3, 480, 640)
        images = forward_inputs["images"].to(self.device, dtype=self.torch_dtype)
        # action shape: 展平格式 (B, chunk_size * action_dim) = (1, 50*14) = (1, 700)
        # 需要 reshape 回 (B, chunk_size, action_dim) = (1, 50, 14)
        actions = forward_inputs["action"].to(self.device, dtype=self.torch_dtype)
        actions = actions.reshape(-1, self.num_action_chunks, self.action_dim)
        # 现在 actions shape: (B, 50, 14)

        # ---- 步骤 1: ACT 前向传播，得到动作分布的 mean ----
        # mean shape: (B, num_action_chunks, action_dim) = (1, 50, 14)
        mean = self._predict_normalized_actions(qpos, images)

        # ---- 步骤 2: 构造动作分布 ----
        # Normal(mean, std=exp(logstd))
        # 这是"新策略"（当前参数）下的动作分布
        distribution = self._action_distribution(mean)

        # ---- 步骤 3: 计算"当时采样的动作"在当前分布下的对数概率 ----
        # distribution.log_prob(actions) shape: (B, 50, 14)
        # .reshape → (B, 700) 展平返回
        result = {"logprobs": distribution.log_prob(actions).reshape(actions.shape[0], -1)}

        # ---- 可选：计算熵（衡量策略的随机程度，用于监控） ----
        if compute_entropy:
            # entropy shape: (B, 50, 14) → reshape 到 (B, 700)
            result["entropy"] = distribution.entropy().reshape(actions.shape[0], -1)
        return result

    # =========================================================================
    # =========================================================================
    # predict_action_batch — 推理时调用：给定观测，输出动作 ★ 核心函数 ★
    # =========================================================================
    # =========================================================================
    # 这个函数在 Rollout Worker 的推理阶段被调用，也在评估时被调用。
    #
    # 完整数据流：
    #
    #   env_obs = {
    #     "states":      (B, 14)    关节位置（原始弧度值）
    #     "main_images":  (B, H, W, 3) 头部相机 RGB 图像 [0-255]
    #     "wrist_images": (B, 2, H, W, 3) 腕部相机图像 [0-255]
    #   }
    #
    #        │
    #        ▼
    #   ① qpos = (states - qpos_mean) / qpos_std    → (B, 14) 归一化
    #   ② images = resize + normalize               → (B, N_cam, 3, H, W)
    #
    #        │ qpos + images
    #        ▼
    #   ③ ACT 模型前向：qpos + images → mean        → (B, 50, 14) 归一化空间的 mean
    #
    #        │ mean
    #        ▼
    #   ④ 构造分布：Normal(mean, std=exp(logstd))
    #
    #        │
    #        ▼
    #   ⑤ 采样动作：
    #       训练时：从分布中随机采样（带噪声）     → exploration
    #       评估时：直接用 mean（无噪声）          → exploitation
    #
    #        │ normalized_actions (B, 50, 14)
    #        ▼
    #   ⑥ 反归一化：action * action_std + action_mean → (B, 50, 14) 真实弧度值
    #
    #        │ env_actions
    #        ▼
    #   ⑦ 返回：
    #       env_actions:     发给 Env 执行的动作
    #       forward_inputs:  保存下来供 Actor 训练时用
    #       prev_logprobs:   旧策略给这个动作的对数概率
    # =========================================================================
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],                                       # 环境观测字典
        mode: str = "eval",                                            # "train" = 采样, "eval" = 确定性
        calculate_logprobs: bool = True,                               # 是否计算 logprob
        calculate_values: bool = False,                                # 是否计算 value（ACT 没有）
        **kwargs,
    ):
        """Predict one ACT action chunk from RoboTwin observations."""

        # ---- 步骤 ①: 归一化关节位置 ----
        # states 原始值可能是弧度 [-3.14, 3.14]
        # 归一化后变成大约 [-1, 1] 范围
        # qpos shape: (B, 14)
        qpos = self._normalize_qpos(env_obs["states"])

        # ---- 步骤 ②: 预处理图像 ----
        # 做三件事：resize、归一化 /255、ImageNet 标准化
        # images shape: (B, N_cam, 3, H, W)
        images = self._prepare_images(env_obs)

        # ---- 步骤 ③: ACT 前向传播 ----
        # mean shape: (B, 50, 14) —— 50 步连续动作，每步 14 维
        mean = self._predict_normalized_actions(qpos, images)

        # ---- 步骤 ④⑤⑥: 根据模式选择采样方式 ----

        # =============================================
        # 情况 A: 评估 + stepwise 模式
        # =============================================
        # "逐步执行"：ACT 一次输出 50 步动作，但执行时每步只取 1 个，
        #            类似控制中的 MPC——每步重新规划但只执行第一步。
        if mode == "eval" and self.eval_stepwise:
            normalized_actions = self._select_eval_step_actions(mean)
            # normalized_actions shape: (B, 1, 14) — 只取 1 步
            distribution = self._action_distribution(normalized_actions)
            # 反归一化：归一化空间的 action → 真实的弧度值
            env_actions = self._denormalize_actions(normalized_actions)
            # 计算 logprob
            logprobs = distribution.log_prob(normalized_actions).reshape(
                normalized_actions.shape[0], -1                         # (B, 14)
            )
            # ACT 没有 value head，全是 0
            values = torch.zeros(
                (normalized_actions.shape[0], 1),
                device=normalized_actions.device,
                dtype=logprobs.dtype,
            )
            # 保存 forward_inputs：Actor 训练时需要"当时看到了什么、采了什么动作"
            forward_inputs = {
                "qpos": qpos.detach().cpu(),                            # (B, 14)
                "images": images.detach().cpu(),                        # (B, N_cam, 3, H, W)
                "action": normalized_actions.detach().cpu().reshape(    # (B, 14)
                    normalized_actions.shape[0], -1
                ),
            }
            result = {
                "prev_logprobs": logprobs.detach().cpu(),
                "prev_values": values.detach().cpu(),
                "forward_inputs": forward_inputs,
            }
            return env_actions.detach().cpu(), result                   # .detach().cpu() 切断梯度，移到 CPU

        # =============================================
        # 情况 B: 训练模式 / 评估+随机模式
        # =============================================
        distribution = self._action_distribution(mean)
        # distribution 覆盖整个 chunk: Normal(mean=(B,50,14), std=(1,50,14))

        if mode == "eval" and not self.use_stochastic_eval:
            # 评估 + 确定性: 直接用 mean，不采样
            normalized_actions = mean                                   # (B, 50, 14)
        else:
            # 训练 OR 评估+随机: 从分布中随机采样（带重参数化技巧）
            # rsample() = mean + std * ε（其中 ε ~ N(0, 1)）
            # 这样采样操作可导，梯度可以穿过
            normalized_actions = distribution.rsample()                 # (B, 50, 14)

        # ---- 步骤 ⑥: 反归一化 ----
        # 把归一化空间的 action 转回机器人可以执行的弧度值
        # env_actions shape: (B, 50, 14)
        env_actions = self._denormalize_actions(normalized_actions)

        # ---- 步骤 ⑦: 计算 logprob 和 forward_inputs ----
        # logprob: 旧策略（采样时刻的策略）对这个 action 的 log 概率
        logprobs = distribution.log_prob(normalized_actions).reshape(
            normalized_actions.shape[0], -1                              # (B, 50*14) = (B, 700)
        )
        # 假的 values（全 0），因为 ACT 没有 Critic/Value Head
        values = torch.zeros(
            (normalized_actions.shape[0], 1),
            device=normalized_actions.device,
            dtype=logprobs.dtype,
        )
        # forward_inputs: 保存下来给 Actor 训练时重新前向用
        forward_inputs = {
            "qpos": qpos.detach().cpu(),                                # (B, 14)
            "images": images.detach().cpu(),                            # (B, N_cam, 3, H, W)
            "action": normalized_actions.detach().cpu().reshape(        # (B, 700)
                normalized_actions.shape[0], -1
            ),
        }
        result = {
            "prev_logprobs": logprobs.detach().cpu(),
            "prev_values": values.detach().cpu(),
            "forward_inputs": forward_inputs,
        }
        return env_actions.detach().cpu(), result

    # =========================================================================
    # reset_eval_state — 重置评估状态
    # =========================================================================
    # 每个评估 episode 开始前调用，清空 stepwise 和 temporal agg 的缓存
    # =========================================================================
    def reset_eval_state(self) -> None:
        """Reset stepwise ACT eval cache between episodes."""
        self._eval_step = 0
        self._eval_action_cache = None
        self._eval_action_history = []

    # =========================================================================
    # device — 属性：返回模型所在的设备（CPU/GPU）
    # =========================================================================
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device                          # 取第一个参数的设备

    # =========================================================================
    # _predict_normalized_actions — 调用 ACT 模型，获取归一化空间的动作均值
    # =========================================================================
    # 这是对 ACT 模型的实际调用。
    # qpos: (B, 14)    归一化后的关节位置
    # images: (B, N_cam, 3, H, W)  预处理后的图像
    # 返回: mean (B, 50, 14)  归一化空间的动作均值
    #
    # ACT 模型内部的数据流（你学过的 CVAE + DETR）：
    #   qpos → MLP → qpos_token
    #   images → ResNet → visual_tokens
    #   visual_tokens + qpos_token → Transformer Encoder → memory
    #   action_queries + memory → Transformer Decoder → action_mean
    #
    # 原始 ACT 返回三个值：(action_mean, encoder_output, decoder_output)
    # 这里只用 action_mean，后两个是 CVAE encoder 的输出，RL 不需要。
    # 只取前 num_action_chunks 步和前 action_dim 维。
    # =========================================================================
    def _predict_normalized_actions(
        self, qpos: torch.Tensor, images: torch.Tensor
    ) -> torch.Tensor:
        # 原始 ACT 模型: qpos + images → (mean, encoder_out, decoder_out)
        action_mean, _, _ = self.act_model(qpos, images, None)
        # action_mean shape: (B, chunk_size, action_dim) = (B, 50, 14)
        return action_mean[:, : self.num_action_chunks, : self.action_dim]

    # =========================================================================
    # _action_distribution — 把 mean 变成 Normal 分布
    # =========================================================================
    # mean: (B, chunk_size, action_dim) = (B, 50, 14)
    # logstd: (1, 50, 14) 可学习参数
    #
    # 做的事：
    #   1. 把 logstd 截断到 [min_logstd, max_logstd]
    #   2. exp(logstd) → std（保证 > 0）
    #   3. 返回 Normal(mean, std)
    #
    # 返回的 distribution 可以用：
    #   distribution.sample()    → 随机采样动作（训练时用）
    #   distribution.rsample()   → 重参数化采样（可导，RL 训练用）
    #   distribution.log_prob(a) → 某个动作的对数概率
    #   distribution.entropy()   → 分布的熵（衡量不确定性）
    #
    # 类比：这就像是给 ACT 的确定性输出加了一个"高斯噪声通道"。
    #       mean = 标称值，std = 噪声强度。
    #       训练时：注入噪声 → 探索。
    #       评估时：零噪声 → 执行。
    # =========================================================================
    def _action_distribution(self, mean: torch.Tensor) -> torch.distributions.Normal:
        # logstd shape: (1, 50, 14)
        # clamp: 限制范围 → exp → std
        logstd = self.logstd.clamp(self.min_logstd, self.max_logstd).to(
            device=mean.device, dtype=mean.dtype
        )
        # 如果 logstd 的第 1 维和 mean 的第 1 维不匹配（比如 stepwise 模式只有 1 步）
        # 就用第 0 步的 std 代替
        if logstd.shape[1] != mean.shape[1]:
            logstd = logstd[:, :1]                                     # (1, 1, 14)
        # std.expand_as(mean): 把 (1, 50, 14) 广播到 (B, 50, 14)
        return torch.distributions.Normal(mean, logstd.exp().expand_as(mean))

    # =========================================================================
    # _select_eval_step_actions — stepwise 评估模式：从 chunk 中取一步
    # =========================================================================
    # ACT 一次输出 50 步动作。stepwise 模式下每步只取 1 个执行。
    # 两种子模式：
    #   1. temporal_agg=True:  用指数加权平均平滑多帧预测（减少抖动）
    #   2. temporal_agg=False: 按顺序从缓存的 chunk 中逐帧取
    # =========================================================================
    def _select_eval_step_actions(self, mean: torch.Tensor) -> torch.Tensor:
        if self.temporal_agg:
            action = self._select_temporal_agg_action(mean)             # 时间聚合模式
        else:
            # 简单 stepwise：缓存整个 chunk，每步取一帧
            # 当缓存为空、步数整除 chunk_size、或 batch 大小变化时，重新缓存
            if (
                self._eval_action_cache is None
                or self._eval_step % self.num_action_chunks == 0
                or self._eval_action_cache.shape[0] != mean.shape[0]
            ):
                self._eval_action_cache = mean                         # 缓存新 chunk
            chunk_index = self._eval_step % self.num_action_chunks    # 当前该取 chunk 的第几帧
            action = self._eval_action_cache[:, chunk_index : chunk_index + 1]  # (B, 1, 14)

        self._eval_step += 1                                           # 步计数器 +1
        return action                                                   # (B, 1, 14)

    # =========================================================================
    # _select_temporal_agg_action — 时间聚合：用指数衰减平滑多帧预测
    # =========================================================================
    # 原理：同一个时间步会被多个 chunk 预测（因为 chunk 有重叠）。
    # 例如 step 0 被 chunk 0 的第 0 帧预测、也可能被 chunk -1 的第 1 帧预测...
    # 对这些"对同一时间步的多个预测"做指数加权平均，权重 = exp(-decay * 时间差)。
    #
    # 类比控制理论：这是一个指数衰减的低通滤波器，去掉动作序列的高频抖动。
    # 越近的预测权重越大，越远的预测权重越小。
    # =========================================================================
    def _select_temporal_agg_action(self, mean: torch.Tensor) -> torch.Tensor:
        self._eval_action_history.append(mean)                         # 保存历史

        # 找到所有包含当前步的 chunk
        start_query = max(0, self._eval_step - self.num_action_chunks + 1)
        action_candidates = []
        for query_step in range(start_query, self._eval_step + 1):
            chunk_index = self._eval_step - query_step                # 这个 chunk 里当前步对应的帧
            action_candidates.append(
                self._eval_action_history[query_step][:, chunk_index, :]  # (B, 14)
            )

        # stacked shape: (B, N_candidates, 14)
        stacked_actions = torch.stack(action_candidates, dim=1)

        # 指数衰减权重: w_i = exp(-decay * i)，i 越大 = 预测越旧 → 权重越小
        weights = torch.exp(
            -self.temporal_agg_decay
            * torch.arange(
                stacked_actions.shape[1],
                device=stacked_actions.device,
                dtype=stacked_actions.dtype,
            )
        )
        weights = weights / weights.sum()                              # 归一化，使权重和为 1

        # 加权平均: (B, N, 14) → (B, 1, 14)
        action = (stacked_actions * weights.view(1, -1, 1)).sum(dim=1, keepdim=True)
        return action

    # =========================================================================
    # _normalize_qpos — 关节位置归一化
    # =========================================================================
    # states: 环境原始观测，可能是 numpy array 或 torch tensor
    # 公式: qpos_normalized = (qpos - qpos_mean) / qpos_std
    #
    # 只取前 state_dim 维（可能有额外的维度如 gripper 状态）
    # qpos_mean/std shape: (1, 14), states shape: (B, 14+)
    # 结果 shape: (B, 14)
    # =========================================================================
    def _normalize_qpos(self, states: Any) -> torch.Tensor:
        states_tensor = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        states_tensor = states_tensor[..., : self.state_dim]           # 截取前 14 维
        # 广播: (B, 14) - (1, 14) / (1, 14) = (B, 14)
        return ((states_tensor - self.qpos_mean) / self.qpos_std).to(self.torch_dtype)

    # =========================================================================
    # _denormalize_actions — 动作反归一化
    # =========================================================================
    # 把归一化空间的 action 转回真实的弧度值。
    # 公式: action_raw = action_normalized * action_std + action_mean
    #
    # normalized_actions shape: (B, chunk_size, action_dim) = (B, 50, 14)
    # action_mean/std shape: (1, 1, 14) → 广播到 (B, 50, 14)
    # 结果 shape: (B, 50, 14) — 真正发给机器人的弧度值
    # =========================================================================
    def _denormalize_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        return normalized_actions * self.action_std + self.action_mean

    # =========================================================================
    # _prepare_images — 图像预处理流水线
    # =========================================================================
    # 处理步骤：
    #   1. 提取头部相机图像    main_images:  (B, H, W, 3) → (B, 3, H, W)
    #   2. 提取腕部相机图像    wrist_images: (B, 2, H, W, 3) → (B, 2, 3, H, W)
    #   3. 按 camera_names 顺序拼接 → (B, N_cam, 3, H, W)
    #   4. 可选 ImageNet 标准化
    #   5. 可选 Resize
    #
    # 最终输出 shape: (B, N_cam, 3, H, W)
    # 例如: (1, 3, 3, 480, 640) = 1个样本, 3个相机, 3通道RGB, 480高, 640宽
    # =========================================================================
    def _prepare_images(self, env_obs: dict[str, Any]) -> torch.Tensor:
        # 头部相机: (B, H, W, 3) → (B, 3, H_resized, W_resized)
        main_images = self._image_tensor(env_obs["main_images"])
        # 腕部相机: (B, 2, H, W, 3) → (B, 2, 3, H_resized, W_resized) 或 None
        wrist_images = env_obs.get("wrist_images")
        wrist_tensor = None if wrist_images is None else self._wrist_tensor(wrist_images)

        # 按 camera_names 的顺序拼接图像
        # 例如: cam_high → cam_left_wrist → cam_right_wrist
        camera_tensors = []
        for camera_name in self.camera_names:
            if camera_name == "cam_high":
                camera_tensors.append(main_images)                     # (B, 3, H, W)
            elif camera_name == "cam_left_wrist":
                if wrist_tensor is None or wrist_tensor.shape[1] < 1:
                    raise ValueError("ACT config requires cam_left_wrist but obs lacks it.")
                camera_tensors.append(wrist_tensor[:, 0])              # (B, 3, H, W) — 第 0 个腕相机
            elif camera_name == "cam_right_wrist":
                if wrist_tensor is None or wrist_tensor.shape[1] < 2:
                    raise ValueError("ACT config requires cam_right_wrist but obs lacks it.")
                camera_tensors.append(wrist_tensor[:, 1])              # (B, 3, H, W) — 第 1 个腕相机
            else:
                raise ValueError(f"Unsupported ACT camera name: {camera_name}")

        # torch.stack 在 dim=1 上拼接: [(B,3,H,W), (B,3,H,W), (B,3,H,W)] → (B, N_cam, 3, H, W)
        images = torch.stack(camera_tensors, dim=1)
        batch_size, num_cameras, channels, height, width = images.shape

        # ImageNet 标准化: (x - mean) / std
        # 把 (B, N_cam, 3, H, W) 展平成 (B*N_cam, 3, H, W) 再标准化，然后恢复
        images = images.reshape(batch_size * num_cameras, channels, height, width)
        if self.image_normalization == "imagenet":
            images = (images - self.image_mean) / self.image_std
        return images.reshape(batch_size, num_cameras, channels, height, width).to(
            self.torch_dtype
        )
        # 最终: (B, N_cam, 3, H, W)

    # =========================================================================
    # _image_tensor — 预处理单组图像（头部相机）
    # =========================================================================
    # 输入: (B, H, W, 3) uint8 [0, 255]
    # 处理: 取前 3 通道 → /255 → HWC→CHW → resize
    # 输出: (B, 3, H', W') float32 [0, 1]
    # =========================================================================
    def _image_tensor(self, images: Any) -> torch.Tensor:
        tensor = torch.as_tensor(images, device=self.device, dtype=torch.float32)
        if tensor.ndim != 4:                                           # 必须是 4 维 (B, H, W, C)
            raise ValueError(f"Expected image tensor [B,H,W,C], got {tuple(tensor.shape)}")
        tensor = tensor[..., :3]                                       # 只取 RGB 通道（丢掉可能的 alpha）
        if tensor.max() > 2.0:                                         # 如果值 > 2.0，说明是 [0,255]，需要除 255
            tensor = tensor / 255.0
        # (B, H, W, 3) → (B, 3, H, W): PyTorch 的标准图像格式
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
        return self._resize_image_tensor(tensor)                       # (B, 3, H', W')

    # =========================================================================
    # _wrist_tensor — 预处理腕部图像（多相机）
    # =========================================================================
    # 输入: (B, N_wrist, H, W, 3) uint8 [0, 255]
    # 处理: 取前 3 通道 → /255 → HWC→CHW → resize → 恢复 (B, N, 3, H', W')
    # 输出: (B, N_wrist, 3, H', W') float32
    # =========================================================================
    def _wrist_tensor(self, images: Any) -> torch.Tensor:
        tensor = torch.as_tensor(images, device=self.device, dtype=torch.float32)
        if tensor.ndim != 5:                                           # 必须是 5 维 (B, N, H, W, C)
            raise ValueError(
                f"Expected wrist image tensor [B,N,H,W,C], got {tuple(tensor.shape)}"
            )
        tensor = tensor[..., :3]                                       # 只取 RGB
        if tensor.max() > 2.0:
            tensor = tensor / 255.0
        batch_size, num_cameras = tensor.shape[:2]
        # (B, N, H, W, 3) → (B, N, 3, H, W)
        tensor = tensor.permute(0, 1, 4, 2, 3).contiguous()
        if self.image_resize_hw is None:
            return tensor
        # resize 需要 (B*N, 3, H, W) → resize → 恢复 (B, N, 3, H', W')
        tensor = tensor.reshape(batch_size * num_cameras, 3, *tensor.shape[-2:])
        tensor = self._resize_image_tensor(tensor)
        return tensor.reshape(batch_size, num_cameras, 3, *self.image_resize_hw)

    # =========================================================================
    # _resize_image_tensor — 缩放图像到目标尺寸
    # =========================================================================
    # F.interpolate: PyTorch 的图像缩放函数
    # mode="bilinear": 双线性插值，适合图像
    # =========================================================================
    def _resize_image_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.image_resize_hw is None or tuple(tensor.shape[-2:]) == self.image_resize_hw:
            return tensor                                               # 不需要 resize
        return F.interpolate(
            tensor,
            size=self.image_resize_hw,                                 # (H, W)
            mode="bilinear",
            align_corners=False,
        )
