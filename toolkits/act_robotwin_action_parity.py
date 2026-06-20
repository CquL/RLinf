"""Compare RoboTwin ACT deploy actions with the RLinf ACT adapter.

This is a diagnostic script for checkpoint migration. It builds one RoboTwin eval
environment, reads the same initial observation, and compares the first ACT
action produced by the original RoboTwin deploy wrapper and by RLinf.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir


def _add_path(path: str | Path) -> None:
    expanded = str(Path(path).expanduser().resolve())
    if expanded not in sys.path:
        sys.path.insert(0, expanded)


def _encode_old_deploy_obs(observation: dict[str, Any]) -> dict[str, Any]:
    """Match policy/ACT/deploy_policy.py::encode_obs exactly."""
    head_cam = cv2.resize(
        observation["observation"]["head_camera"]["rgb"],
        (640, 480),
        interpolation=cv2.INTER_LINEAR,
    )
    left_cam = cv2.resize(
        observation["observation"]["left_camera"]["rgb"],
        (640, 480),
        interpolation=cv2.INTER_LINEAR,
    )
    right_cam = cv2.resize(
        observation["observation"]["right_camera"]["rgb"],
        (640, 480),
        interpolation=cv2.INTER_LINEAR,
    )
    qpos = (
        observation["joint_action"]["left_arm"]
        + [observation["joint_action"]["left_gripper"]]
        + observation["joint_action"]["right_arm"]
        + [observation["joint_action"]["right_gripper"]]
    )
    return {
        "head_cam": np.moveaxis(head_cam, -1, 0) / 255.0,
        "left_cam": np.moveaxis(left_cam, -1, 0) / 255.0,
        "right_cam": np.moveaxis(right_cam, -1, 0) / 255.0,
        "qpos": qpos,
    }


def _build_old_args(model_cfg: Any, device: str) -> dict[str, Any]:
    ckpt_dir = str(Path(model_cfg.model_path).expanduser().resolve().parent)
    return {
        "lr": float(model_cfg.get("lr", 1e-5)),
        "lr_backbone": float(model_cfg.get("lr_backbone", 1e-5)),
        "batch_size": int(model_cfg.get("batch_size", 1)),
        "weight_decay": float(model_cfg.get("weight_decay", 1e-4)),
        "epochs": int(model_cfg.get("epochs", 1)),
        "chunk_size": int(model_cfg.get("chunk_size", model_cfg.num_action_chunks)),
        "kl_weight": float(model_cfg.get("kl_weight", 10.0)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 512)),
        "dim_feedforward": int(model_cfg.get("dim_feedforward", 3200)),
        "backbone": str(model_cfg.get("backbone", "resnet18")),
        "enc_layers": int(model_cfg.get("enc_layers", 4)),
        "dec_layers": int(model_cfg.get("dec_layers", 7)),
        "nheads": int(model_cfg.get("nheads", 8)),
        "position_embedding": str(model_cfg.get("position_embedding", "sine")),
        "dilation": bool(model_cfg.get("dilation", False)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "pre_norm": bool(model_cfg.get("pre_norm", False)),
        "masks": bool(model_cfg.get("masks", False)),
        "camera_names": list(model_cfg.camera_names),
        "action_dim": int(model_cfg.get("action_dim", 14)),
        "state_dim": int(model_cfg.get("state_dim", 14)),
        "temporal_agg": bool(model_cfg.get("temporal_agg", True)),
        "ckpt_dir": ckpt_dir,
        "device": device,
    }


def _old_normalized_chunk(old_model: Any, old_obs: dict[str, Any]) -> torch.Tensor:
    qpos_numpy = np.array(old_obs["qpos"])
    qpos = torch.from_numpy(old_model.pre_process(qpos_numpy)).float()
    qpos = qpos.to(old_model.device).unsqueeze(0)
    curr_image = np.stack(
        [old_obs["head_cam"], old_obs["left_cam"], old_obs["right_cam"]],
        axis=0,
    )
    image = torch.from_numpy(curr_image).float().to(old_model.device).unsqueeze(0)
    with torch.no_grad():
        return old_model.policy(qpos, image).detach().cpu()


def _tensor_stats(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    diff = (lhs.float() - rhs.float()).abs()
    print(
        f"{name}: shape={tuple(lhs.shape)} max_abs={diff.max().item():.8f} "
        f"mean_abs={diff.mean().item():.8f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-path",
        default=os.environ.get("REPO_PATH", "/home/lhj/RLinf-main"),
    )
    parser.add_argument(
        "--robotwin-path",
        default=os.environ.get("ROBOTWIN_PATH", "/home/lhj/RoboTwin-RLinf_support"),
    )
    parser.add_argument(
        "--act-source-dir",
        default="/home/lhj/robot_l/robotwin/policy/ACT",
    )
    parser.add_argument(
        "--config-name",
        default="robotwin_beat_block_hammer_eval_act",
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--total-num-processes", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_path = Path(args.repo_path).expanduser().resolve()
    robotwin_path = Path(args.robotwin_path).expanduser().resolve()
    act_source_dir = Path(args.act_source_dir).expanduser().resolve()

    os.environ.setdefault("REPO_PATH", str(repo_path))
    os.environ.setdefault("EMBODIED_PATH", str(repo_path / "examples" / "embodiment"))
    os.environ.setdefault("ROBOTWIN_PATH", str(robotwin_path))
    os.environ.setdefault("ASSETS_PATH", str(robotwin_path))
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    _add_path(repo_path)
    _add_path(robotwin_path)
    _add_path(robotwin_path / "robotwin")
    _add_path(act_source_dir)

    config_dir = repo_path / "examples" / "embodiment" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=args.config_name)

    device = args.device if torch.cuda.is_available() else "cpu"

    from act_policy import ACT  # type: ignore
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.act.act_rl_policy import ACTRLPolicy

    env = RoboTwinEnv(
        cfg.env.eval,
        num_envs=1,
        seed_offset=args.seed_offset,
        total_num_processes=args.total_num_processes,
        worker_info=None,
        record_metrics=False,
    )
    try:
        rlinf_obs, _ = env.reset()
        raw_obs = env.venv.envs[0].task.get_obs()
        old_obs = _encode_old_deploy_obs(raw_obs)

        old_args = _build_old_args(cfg.actor.model, device)
        old_model = ACT(old_args, Namespace(**old_args))
        old_action = old_model.get_action(old_obs)
        old_action_tensor = torch.as_tensor(old_action, dtype=torch.float32).reshape(
            1, 1, -1
        )
        old_chunk = _old_normalized_chunk(old_model, old_obs)

        rlinf_model = ACTRLPolicy(cfg.actor.model).to(device).eval()
        rlinf_model.reset_eval_state()
        with torch.no_grad():
            rlinf_action, _ = rlinf_model.predict_action_batch(rlinf_obs, mode="eval")
            rlinf_qpos = rlinf_model._normalize_qpos(rlinf_obs["states"]).detach().cpu()
            rlinf_images = rlinf_model._prepare_images(rlinf_obs).detach().cpu()
            rlinf_chunk = (
                rlinf_model._predict_normalized_actions(
                    rlinf_qpos.to(device), rlinf_images.to(device)
                )
                .detach()
                .cpu()
            )

        old_qpos = torch.as_tensor(old_model.pre_process(np.array(old_obs["qpos"])))
        rlinf_state = torch.as_tensor(rlinf_obs["states"][0], dtype=torch.float32)

        print(f"reset_state_id={int(env.reset_state_ids[0])}")
        print(
            "raw_image_shapes="
            f"head={raw_obs['observation']['head_camera']['rgb'].shape} "
            f"left={raw_obs['observation']['left_camera']['rgb'].shape} "
            f"right={raw_obs['observation']['right_camera']['rgb'].shape}"
        )
        _tensor_stats("raw_qpos_vs_rlinf_state", torch.as_tensor(old_obs["qpos"]), rlinf_state)
        _tensor_stats("normalized_qpos", old_qpos.reshape(1, -1), rlinf_qpos)
        _tensor_stats("normalized_action_chunk", old_chunk, rlinf_chunk)
        _tensor_stats("first_env_action", old_action_tensor, rlinf_action)
        print("old_first_env_action=", old_action_tensor.flatten().numpy())
        print("rlinf_first_env_action=", rlinf_action.flatten().numpy())
    finally:
        env.close(clear_cache=True)


if __name__ == "__main__":
    main()
