# ACT on RoboTwin with RLinf

This is the local runbook for taking the old RoboTwin ACT checkpoint into
RLinf, validating it with `only_eval: True`, then starting a small GRPO
post-training run.

## 0. What We Validate First

Do not start RL training first. The first artifact is an ACT checkpoint eval in
RLinf's `RoboTwinEnv`:

- config: `examples/embodiment/config/robotwin_beat_block_hammer_eval_act.yaml`
- ACT model config: `examples/embodiment/config/model/act.yaml`
- checkpoint:
  `/home/lhj/robot_l/robotwin/policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50/policy_last.ckpt`
- stats:
  `/home/lhj/robot_l/robotwin/policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50/dataset_stats.pkl`
- RoboTwin support repo: `/home/lhj/RoboTwin-RLinf_support`

Current local check: the ACT checkpoint loads into the declared DETR-ACT
architecture with `missing=0` and `unexpected=0`.

For this ACT checkpoint, keep `center_crop: False`. The old ACT preprocessing
uses full `640x480` RGB observations; cropping to `224x224` is for other
VLA-style policies and changes the ACT input distribution.

Keep `image_normalization: "imagenet"` for this checkpoint. The old ACT
`deploy_policy.py` first scales RGB observations to `[0, 1]`, then
`ACTPolicy.__call__` applies ImageNet mean/std normalization before the ResNet
backbone.

Also keep `image_resize_hw: [480, 640]`. The old ACT deploy path resizes each
RoboTwin camera image from the environment's `240x320` frame to `480x640`
before model inference. Without this resize, qpos parity can be exact while the
predicted action chunk is still wrong.

## 1. Prepare Assets

The current `/home/lhj/RoboTwin-RLinf_support/assets` directory only contains
the downloader and images. RoboTwin will fail before reset until
`assets/embodiments` and task objects exist.

On this machine, the old RoboTwin checkout already has the full assets under
`/home/lhj/robot_l/robotwin/assets`. The fastest path is to reuse them with
symlinks:

```bash
cd /home/lhj/RoboTwin-RLinf_support
ln -sfn /home/lhj/robot_l/robotwin/assets/embodiments assets/embodiments
ln -sfn /home/lhj/robot_l/robotwin/assets/objects assets/objects
ln -sfn /home/lhj/robot_l/robotwin/assets/background_texture assets/background_texture
```

Use the download path only if those old assets are unavailable.

```bash
cd /home/lhj/RoboTwin-RLinf_support
bash script/_download_assets.sh
python script/update_embodiment_config_path.py
```

If downloading fails with `Unknown scheme for proxy URL URL('socks://...')`,
stop the prompt with `Ctrl+C`, then rerun with HTTP proxy variables:

```bash
cd /home/lhj/RoboTwin-RLinf_support
unset ALL_PROXY all_proxy
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HF_ENDPOINT=https://hf-mirror.com
python -m pip install -U huggingface_hub
bash script/_download_assets.sh
```

If `huggingface_hub` still fails with metadata errors on the mirror, the local
download script falls back to direct zip downloads. You can also force a base
URL explicitly:

```bash
cd /home/lhj/RoboTwin-RLinf_support
unset ALL_PROXY all_proxy
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export ROBOTWIN_ASSET_BASE_URL=https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main
bash script/_download_assets.sh
```

Acceptance check:

```bash
test -d /home/lhj/RoboTwin-RLinf_support/assets/embodiments
test -f /home/lhj/RoboTwin-RLinf_support/assets/objects/objaverse/list.json
```

In RLinf configs, set `env.*.assets_path` to the RoboTwin support repository
root (`/home/lhj/RoboTwin-RLinf_support`), not to the nested `assets`
directory. RoboTwin appends `assets/...` internally.

## 2. Prepare One Python Environment

Ray workers inherit the Python environment captured at `ray start`, so RLinf
and RoboTwin must be installed in the same environment before Ray starts.

On this machine, use the existing `RoboTwin` conda environment. It already has
CUDA PyTorch and RoboTwin-side dependencies.

```bash
conda activate RoboTwin
cd /home/lhj/RLinf-main
python -m pip install -e .
python -m pip install hydra-core omegaconf "ray[default]" tensorboard regex
```

After that, check the minimum imports:

```bash
python -c "import torch, ray, hydra, omegaconf; print(torch.__version__, torch.cuda.is_available())"
python -c "import sys; sys.path[:0]=['/home/lhj/RLinf-main','/home/lhj/RoboTwin-RLinf_support']; import rlinf, robotwin; print('ok')"
```

## 3. Run The First ACT-RLinf Eval

Start Ray only after the environment and paths are ready. Start it from the
real activated `RoboTwin` shell. Avoid `conda run -n RoboTwin ray start ...`
for this workflow, because the Ray daemons can inherit a short-lived launcher
environment and disappear while the eval process is trying to connect.

```bash
conda activate RoboTwin
cd /home/lhj/RLinf-main
export REPO_PATH=/home/lhj/RLinf-main
export EMBODIED_PATH=/home/lhj/RLinf-main/examples/embodiment
export ROBOTWIN_PATH=/home/lhj/RoboTwin-RLinf_support
export ASSETS_PATH=/home/lhj/RoboTwin-RLinf_support
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH=/home/lhj/RLinf-main:/home/lhj/RoboTwin-RLinf_support:$PYTHONPATH
export RLINF_NODE_RANK=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
ray stop
ray start --head --port=6379 --disable-usage-stats
ray status
python examples/embodiment/eval_embodied_agent.py \
  --config-path /home/lhj/RLinf-main/examples/embodiment/config \
  --config-name robotwin_beat_block_hammer_eval_act
```

Before the full 400-step eval, a fast smoke run can verify reset, image crop,
ACT action generation, and one 50-action chunk:

```bash
python examples/embodiment/eval_embodied_agent.py \
  --config-path /home/lhj/RLinf-main/examples/embodiment/config \
  --config-name robotwin_beat_block_hammer_eval_act \
  env.eval.max_episode_steps=50 \
  env.eval.max_steps_per_rollout_epoch=50 \
  env.train.max_episode_steps=50 \
  env.train.max_steps_per_rollout_epoch=50 \
  env.eval.task_config.step_lim=50 \
  env.train.task_config.step_lim=50
```

Acceptance check:

- the job reaches RoboTwin reset without import or asset errors
- `eval/success_once`, `eval/return`, and episode metrics appear in the log
- output log is under `logs/<timestamp>-robotwin_beat_block_hammer_eval_act/`

If this eval is much worse than the old ACT project, compare against the old
project with the same conditions first: clean domain, 400 step limit, same
checkpoint, and no temporal aggregation unless RLinf also implements it.

Known old-project ACT eval references for this checkpoint/task are:

- `2026-05-18 10:58:42`: `0.69`
- `2026-05-22 14:55:29`: `0.72`

Current local ACT baseline eval note on 2026-06-15: running
`robotwin_beat_block_hammer_eval_act` with `runner.ckpt_path=null`,
`env.eval.total_num_envs=2`, and `algorithm.eval_rollout_epoch=10` completed
successfully. The config loaded the old ACT
`demo_clean-50/policy_best.ckpt` directly. The result was
`eval/success_once=0.0`, `eval/success_at_end=0.0`, `eval/return=0.0`, and
`eval/num_trajectories=20`, so the RLinf ACT baseline currently measures
`0/20 = 0%`. Because the old ACT project has nonzero clean eval results, treat
this as an ACT adapter / RoboTwin config parity issue before continuing GRPO
post-training.

This early `0/20` is now understood as a parity problem, not checkpoint
corruption. The fixes are: load the old deploy-compatible `policy_last.ckpt`,
resize images to `480x640`, run ACT eval with `eval_stepwise: true` plus
`temporal_agg: true`, and set `env.eval.step_mode: take_action`.

Do not disable ImageNet normalization as a fix for this `0/20` result. A
follow-up check with `image_normalization: "none"` also measured `0/20`, and
the old deployment wrapper does use ImageNet mean/std inside
`ACTPolicy.__call__`.

The ACT eval adapter should keep the same camera order as the old
`deploy_policy.py` input path: head camera, left wrist camera, then right wrist
camera. In RLinf config this is
`camera_names: ["cam_high", "cam_left_wrist", "cam_right_wrist"]`.

Parity fix applied after this diagnosis: ACT eval now supports
`eval_stepwise: true` plus `temporal_agg: true`. In this mode RLinf computes
400 eval steps instead of `400 / 50` chunk steps, resets the ACT eval cache at
the start of each eval rollout, re-predicts a 50-step chunk each sim step, and
sends only the aggregated 1-step action to RoboTwin. The ACT eval config also
sets `env.eval.step_mode: take_action`, so RoboTwin executes that action with
the same `TASK_ENV.take_action(action)` path as the old deploy script. Training
rollout remains 50-step chunk based and continues to use
`gen_sparse_reward_data(chunk_actions)`.

Current local checkpoint eval note on 2026-06-15: evaluating
`global_step_50/actor/model_state_dict/full_weights.pt` with
`env.eval.total_num_envs=20` failed before any success metric was produced.
The run loaded the config and workers, then crashed during
`RoboTwinEnv.reset()` while camera observations were being created:
`RuntimeError: cannot create buffer` from
`envs/camera/camera.py::take_picture`. This was preceded by repeated
`svulkan2` `OIDN Error: unsupported device type: CUDA` /
`OIDN Error: invalid handle` messages. Treat this as an eval environment
concurrency/render-buffer failure, not as a measured 0% eval result.

On the 8GB laptop GPU, run eval with low environment concurrency and use more
rollout epochs to collect more episodes:

```bash
python examples/embodiment/eval_embodied_agent.py \
  --config-path /home/lhj/RLinf-main/examples/embodiment/config \
  --config-name robotwin_beat_block_hammer_eval_act \
  runner.ckpt_path=/home/lhj/results/robotwin_beat_block_hammer_act_eval/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt \
  env.eval.total_num_envs=2 \
  algorithm.eval_rollout_epoch=10
```

For the most conservative smoke eval, start with one environment:

```bash
python examples/embodiment/eval_embodied_agent.py \
  --config-path /home/lhj/RLinf-main/examples/embodiment/config \
  --config-name robotwin_beat_block_hammer_eval_act \
  runner.ckpt_path=/home/lhj/results/robotwin_beat_block_hammer_act_eval/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt \
  env.eval.total_num_envs=1 \
  algorithm.eval_rollout_epoch=5
```

Follow-up eval with low concurrency completed successfully:

```bash
python examples/embodiment/eval_embodied_agent.py \
  --config-path /home/lhj/RLinf-main/examples/embodiment/config \
  --config-name robotwin_beat_block_hammer_eval_act \
  runner.ckpt_path=/home/lhj/results/robotwin_beat_block_hammer_act_eval/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt \
  env.eval.total_num_envs=2 \
  algorithm.eval_rollout_epoch=10
```

The result was `eval/success_once=0.0`, `eval/success_at_end=0.0`,
`eval/return=0.0`, and `eval/num_trajectories=20`, so `global_step_50` measured
`0/20 = 0%` on this deterministic clean eval. The repeated `svulkan2` OIDN
messages still appeared, but unlike the 20-env attempt they did not stop the
run.

After the parity fixes, the raw ACT baseline is no longer at zero in RLinf:
a one-env smoke eval reached `eval/success_once=1.0`, and a 5-env low-concurrency
eval reached `eval/success_once=0.6` / `eval/success_at_end=0.6` over
`eval/num_trajectories=5`. Treat this as a smoke result; the next formal
baseline should use 100 distinct eval seeds collected in low-concurrency
batches.

## 4. Start Small GRPO Post-Training

Only after eval is stable, launch a tiny run. The command below keeps the same
config and overrides the minimum training knobs. On an 8GB single GPU, keep
training at 2 envs first and leave both actor and rollout offload enabled;
otherwise the actor-to-rollout weight sync or the next rollout model reload can
run out of CUDA memory.

```bash
conda activate RoboTwin
cd /home/lhj/RLinf-main
export REPO_PATH=/home/lhj/RLinf-main
export EMBODIED_PATH=/home/lhj/RLinf-main/examples/embodiment
export ROBOTWIN_PATH=/home/lhj/RoboTwin-RLinf_support
export ASSETS_PATH=/home/lhj/RoboTwin-RLinf_support
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH=/home/lhj/RLinf-main:/home/lhj/RoboTwin-RLinf_support:$PYTHONPATH
export RLINF_NODE_RANK=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ray stop
ray start --head --port=6379 --disable-usage-stats
python examples/embodiment/train_embodied_agent.py \
  --config-path /home/lhj/RLinf-main/examples/embodiment/config \
  --config-name robotwin_beat_block_hammer_eval_act \
  runner.only_eval=False \
  runner.max_epochs=10 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  algorithm.group_size=2 \
  env.train.total_num_envs=2 \
  env.train.group_size=2 \
  env.eval.total_num_envs=1 \
  env.train.video_cfg.save_video=False \
  actor.micro_batch_size=1 \
  actor.global_batch_size=2 \
  actor.enable_offload=True \
  rollout.enable_offload=True
```

First-run acceptance check:

- the run passes at least 4 global steps, because the previous failing point
  was the weight sync after step 3
- `success_once`, `advantages_max/min`, `actor/grad_norm`, and
  `actor/policy_loss` appear in the metric table
- if that is stable, rerun with `runner.val_check_interval=5` and
  `runner.save_interval=5`

Current local smoke result on 2026-06-15: with `rollout.enable_offload=True`,
`env.train.total_num_envs=2`, `algorithm.group_size=2`, and
`runner.max_epochs=4`, training reached `Global Step: 4/4` without CUDA OOM.
The step-4 table showed `success_once=0.5`, `advantages_max=0.707`,
`advantages_min=-0.707`, `actor/grad_norm=87.349`, and
`actor/policy_loss=0.191`, so the GRPO update path is active.

Follow-up 10-step run reached step 5 with valid GRPO signal, then failed while
`rollout.reload_model()` moved the rollout model back to GPU. The GPU had only
14.94 MiB free because the actor still held about 3.11 GiB and the env held
about 2.62 GiB. Keep `actor.enable_offload=True` for this 8GB setup.

After enabling actor offload, the same 10-step command reached
`Global Step: 10/10` without CUDA OOM. It passed the previous failing reload
point after step 5. The run showed real GRPO updates on successful mixed
groups, for example step 3 had `success_once=0.5`, `advantages_max=0.707`,
`advantages_min=-0.707`, `actor/grad_norm=113.3`, and
`actor/policy_loss=0.227`; step 5 and step 6 also had non-zero advantages and
gradients.

A later 50-step run with `env.train.total_num_envs=2`,
`env.train.group_size=2`, `actor.global_batch_size=2`, actor/rollout offload,
and `runner.save_interval=10` reached `Global Step: 50/50` and saved
`../results/robotwin_beat_block_hammer_act_eval/checkpoints/global_step_50`.
This proves the longer single-GPU run can finish, but it does not prove every
step produced a learning signal. The last logged steps showed groups with the
same outcome for both trajectories: step 47 had `success_once=1.0` with
`advantages_max=0.0000`, while steps 48-50 had `success_once=0.0` and
`advantages_max=0.0000`. For GRPO this is expected: if every trajectory in a
group has the same score, the group-relative advantage is zero and
`actor/policy_loss`, `actor/grad_norm`, and `actor/total_loss` can all be zero.

Repeated `svulkan2` lines like `OIDN Error: invalid handle` appeared during
that run, but they were not the blocker for training because rollout continued,
checkpointing completed, and the metric table was printed. Treat them as a
renderer/denoiser warning unless video capture or rendered observations become
blank or corrupted.

## 5. Scale Only After A Stable Small Run

Scale in this order:

1. increase `env.train.total_num_envs` from 4 to 8 or 16
2. keep `algorithm.group_size` at 4 first
3. increase `actor.global_batch_size` to match collected rollout size
4. enable randomized domain only after clean eval and clean GRPO improve or hold

For randomized training, first change only these fields:

```yaml
env:
  train:
    task_config:
      domain_randomization:
        random_background: true
        cluttered_table: true
        random_light: true
```

Keep eval clean and randomized as two separate configs so success-rate movement
is easy to interpret.
