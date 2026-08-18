---
name: so101-train-policy
description: LeRobot 0.5.2에서 SO-101 ACT/SmolVLA 데이터셋을 검사하고 학습 명령, LoRA/PEFT 설정, W&B 기록, checkpoint 검증, 재개·출력 경로 전략을 만든다. Use for training, fine-tuning, LoRA, PEFT, batch size, steps, learning rate, W&B, checkpoint, or Hugging Face model questions.
---

# SO-101 Train Policy

## Load context

Read:

1. `project/docs/agent-context/project-contract.md`
2. `project/docs/agent-context/dataset-model-registry.md`
3. `project/docs/agent-context/experiment-history.md`
4. `project/config/experiment-profiles/active.env`

Read `decisions-and-known-failures.md` when reproducing a failed command or
comparing preprocessing changes.

## Never guess this checkout's CLI

This repository is LeRobot `0.5.2`. Before writing an exact command:

```bash
cd "${LEROBOT_ROOT:-$HOME/lerobot}"
conda activate lerobot
lerobot-train --help
```

Reject unsupported copied flags. Known failures include:

- `--dataset.image_transforms.tfs.*`
- `--eval_freq=0`

Use the flag names displayed by the local help and current dataclasses. If the
requested transform cannot be expressed in this version, say so instead of
inventing syntax.

## Gate training on dataset inspection

Verify before allocating the GPU:

1. Dataset exists at the intended Hub ID or local root.
2. Episode count, task values, FPS, robot type, and video files are complete.
3. State/action feature names, order, shape, dtype, and stats are sensible.
4. All three cameras decode and belly orientation is correct.
5. Dataset image keys match the requested rename strategy.
6. Train/validation split or the lack of one is explicit.
7. Relabeled data still describes the recorded behavior truthfully.
8. Recovery and nominal examples are balanced for the intended comparison.

For `top/wrist/belly` datasets targeting a policy that expects
`camera1/camera2/camera3`, inspect whether the training command needs a rename
map and whether that rename processor is saved with the checkpoint. Never apply
two competing rename maps.

## Choose the training family deliberately

- Use LoRA/PEFT when preserving the base vision-language backbone and iterating
  quickly on a modest dataset.
- Use full fine-tuning only for an explicit comparison with adequate compute,
  data diversity, and overfitting controls.
- Do not call `freeze_vision_encoder=true` universally correct. It is a current
  LoRA reference, not a law.
- Do not choose learning rate from the run name alone. Historical runs used
  both `1e-4` and `1e-3` for different experiments.

Historical later-run references:

- base `lerobot/smolvla_base`;
- LoRA rank/alpha `64/64`;
- batch 8;
- warmup 1000;
- AdamW beta `(0.9, 0.95)`, weight decay `0.01`;
- cosine decay;
- `freeze_vision_encoder=true`;
- `train_expert_only=true`;
- `train_state_proj=true`;
- `use_amp=false`.

Treat each as a candidate to verify against the exact run, dataset size, and
local help.

## Build the command safely

1. Source the active profile and print it without secrets.
2. Require a non-empty learning rate and an explicit dataset/model goal.
3. Choose a unique `RUN_NAME` and output directory.
4. Refuse to delete an existing output directory. Use a new name or an
   officially supported resume option.
5. Add `CUDA_VISIBLE_DEVICES` and
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only at process launch.
6. Include complete stdout/stderr logging and W&B only when configured.
7. Push dataset/model artifacts only if the user requests publication.
8. Echo the full effective command before starting.

When the user asks only for a command, still perform the read-only checks and
return one complete copy-pasteable command rather than disconnected fragments.

## Smoke before the full run

Run a small unique smoke job when execution is authorized:

- a few steps;
- same dataset feature route and model family;
- separate output directory and W&B run;
- verify first batch, loss, gradients, GPU memory, save, and reload.

Do not infer success from process uptime alone. Check the log and resulting
artifact.

## Monitor and interpret

Record:

- exact command and environment overrides;
- dataset ID/root and model/base ID;
- Git commit and dirty-file summary;
- W&B run URL/ID;
- loss, gradient norm, learning rate, throughput, GPU memory;
- checkpoint timestamps and sizes;
- any worker/video decoder warnings.

Training loss alone does not establish physical-policy quality. Compare
rollouts using `so101-evaluate-experiment`.

## Validate a checkpoint before robot inference

Inspect and smoke-load:

- policy config;
- weights or full merged model files;
- `adapter_config.json` and adapter weights when PEFT is used;
- preprocessor/postprocessor config;
- dataset stats/normalization state;
- image/state/action feature names;
- saved rename map;
- task/language processor files.

Distinguish adapter-only, full, and merged checkpoints. A missing
`adapter_config.json` or full-weight file must be resolved on the GPU without
connecting the robot.

## Update project state

After verified training:

1. Add/update the registry row with exact parameters and status.
2. Do not set `active` until physical evaluation evidence exists.
3. Update `active.env` only after the user selects the model/profile.
4. Preserve previous model IDs for reproducible comparisons.
