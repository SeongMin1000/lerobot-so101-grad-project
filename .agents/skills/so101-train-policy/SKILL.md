---
name: so101-train-policy
description: LeRobot 0.5.2에서 SO-101 ACT 및 SmolVLA 정책 모델을 학습·파인튜닝하고, Steps/Epochs 계산, Loss 지표 분석, W&B 기록, Hub 배포, 이어서 학습(Resume vs Pretrained Continuation) 전략을 수행한다. Use for training, fine-tuning, ACT, SmolVLA, LoRA, batch size, steps, epochs, L1 loss, resume, W&B, checkpoint, or Hugging Face model questions.
---

# SO-101 Train Policy

## 1. Load context & LeRobot 0.5.2 CLI Constraints

Read before acting:

1. `project/docs/agent-context/project-contract.md`
2. `project/docs/agent-context/dataset-model-registry.md`
3. `project/docs/agent-context/hardware-and-paths.md`
4. `project/config/experiment-profiles/active.env`

### CLI 규칙 (LeRobot 0.5.2)
반드시 현재 설치된 `0.5.2` 버전의 CLI 옵션을 준수해야 하며, 타 버전 플래그 복사 사용을 금지합니다:
- 거부되는 미지원 플래그: `--dataset.image_transforms.tfs.*`, `--eval_freq=0`
- 점검 명령: `lerobot-train --help`

---

## 2. Dataset Preflight & 2-Camera Key Contract

학습 시작 전 데이터셋 무결성 검증:
1. 데이터셋 경로 또는 Hugging Face Hub ID가 유효하고 로드 가능한지 확인.
2. 에피소드 수, 태스크 텍스트, 30 FPS 타임스탬프, 비디오 디코딩 정상 여부 확인.
3. **2-Camera Key Contract**:
   - 데이터셋의 `top`, `wrist` 영상이 모델의 `camera1`, `camera2`로 일관되게 매핑되는지 확인.
   - 단일 Rename Processor(`top -> camera1, wrist -> camera2`)가 체크포인트와 함께 저장되는지 확인.
4. 관절 각도(`observation.state`) 및 액션(`action`) 피처의 차원, 단위, 정규화 통계(stats) 확인.

### 2.1 [필수] 이미지 증강 (Image Augmentation) 기본 적용
실제 환경의 조명 변화, 그림자, 카메라 시야각 편차에 대한 정책 강인성(Robustness)을 위해 **모든 모델 학습 시 이미지 증강을 무조건 활성화**합니다:
- 플래그: `--dataset.image_transforms.enable=true --dataset.image_transforms.max_num_transforms=3`
- 내장 증강 내용: Brightness (0.8~1.2), Contrast (0.8~1.2), Saturation (0.5~1.5), Hue (-0.05~0.05), Sharpness (0.5~1.5), RandomAffine (±5°, translate ±5%).

### 2.2 [필수] Train / Validation 분할 및 검증(Val Loss) 규칙
모델의 과적합(Overfitting)을 조기에 감지하고 최적의 일반화 체크포인트를 선별하기 위해 **데이터셋 분할 및 검증을 필수로 적용**합니다:

1. **분할 비율 가이드**:
   - **대규모 데이터셋 (200~300+ 에피소드)**: **Train 90~95% / Validation 5~10%** 분할 필수 적용 (학습 데이터 다양성을 유지하면서 미학습 검증 세트 확보).
   - **소규모 데이터셋 (<100 에피소드)**: 데이터 누락 방지를 위해 100% 학습을 우선하되, 필요 시 95:5 최소 분할 적용.
2. **검증 손실(Val Loss)의 오프라인 채점 원리**:
   - 시뮬레이터를 켜서 잡았는지 시험하는 것이 아니라, **"녹화된 Val 에피소드의 카메라 영상 보고 모델이 예측한 액션"**과 **"사람이 조종했던 실제 정답 액션"** 사이의 오차를 수학적으로 채점하여 W&B에 기록합니다.
3. **골든 체크포인트(Golden Checkpoint) 선별법**:
   - Train Loss는 계속 떨어지는데 Val Loss가 반등하거나 정체되는 시점이 과적합(단순 암기) 시작점입니다.
   - W&B에서 **Val Loss가 가장 낮게 기록된 체크포인트**를 실제 로봇 Rollout 평가의 **1순위 최적 모델**로 선정합니다.

---

## 3. Training Family A: ACT (Action Chunking with Transformers)

단일 태스크 고속 수렴, 5블록 정밀 조작 및 실시간 추론에 최적화된 정책.

### 3.1 공식 권장 하이퍼파라미터 (기본값)
- **Batch Size**: 16 (RTX 3090 24GB 기준 VRAM ~1GB 미만으로 매우 가벼움)
- **Steps**: **100,000 ~ 150,000 steps** (기본 10만~15만 스텝 권장)
- **Optimizer**: AdamW, Learning Rate **`1e-4`** (또는 `1e-5`), Weight Decay `1e-4`
- **Action Chunk**: `n_action_steps=100`
- **체크포인트 저장 주기**: `save_freq=10000` (1만 스텝마다 저장)
- **이미지 증강**: `--dataset.image_transforms.enable=true` (기본 활성화)

### 3.2 Steps ↔ Epochs 계산 법칙 (과소적합 방지)
로봇 모방 학습은 수십만 스텝이라는 단순 수치가 아닌 **"데이터셋을 몇 번 반복(Epoch) 학습했는가"**를 기준으로 판단해야 합니다.

$$\text{1 Epoch Steps} = \left\lceil \frac{\text{Total Frames}}{\text{Batch Size}} \right\rceil, \quad \text{Total Steps} = \text{Target Epochs} \times \text{1 Epoch Steps}$$

* **실제 대규모 데이터셋 예시**: 약 30만 프레임(200+ 에피소드), Batch Size 16 기준:
  * $1\text{ Epoch} \approx 18,955\text{ steps}$
  * $50,000\text{ steps} \approx 2.64\text{ epochs}$ (데이터셋을 겨우 2.6회 본 수준)
* **주의**: 5만 스텝 수준에서 Loss 감소가 둔화되어 평평해 보이더라도, 이는 평균 궤적만 학습된 초기 단계이며 수렴이 아닙니다. **미세 파지 오차(1~2cm)를 줄이려면 최소 5~8 Epochs (100k ~ 150k+ steps) 학습이 필수적**입니다.

### 3.3 ACT Loss 지표 해석 가이드 (👉 `L1 Loss` 집중 모니터링)
ACT는 CVAE 회귀 방식이므로 Total Loss에 속지 말고 **`L1 Loss`**를 Log Scale로 분석해야 합니다:

1. **`L1 Loss` (핵심 모니터링 지표)**:
   - 실제 시연 action과 모델의 예측 action 간 평균 절대 오차 ($|\text{사람 정답} - \text{모델 예측}|$).
   - L1 Loss가 낮을수록 로봇의 파지/안착 위치 정확도가 향상됨.
   - 예: L1 Loss가 `0.067` 수준이면 블록 방향 판단과 슬롯 이동은 잘 수행하지만, 마지막 블록 접근 시 **1~2cm의 미세 접근 오차(Grasp miss)**가 발생할 수 있음 $\to$ 추가 Epoch 학습으로 L1 Loss를 더 낮춰야 함.
2. **`KLD Loss` (VAE 잠재 공간 지표)**:
   - VAE latent representation의 발산/수렴 지표.
   - 0.0001 수준으로 매우 낮아도(Posterior Collapse), ACT 추론 시에는 일반적으로 prior를 기반으로 action을 생성하므로 롤아웃 성능에 치명적이지 않음.
3. **`Total Loss`**:
   - Total Loss = L1 Loss + KLD Loss. 두 값의 스케일 차이로 인해 Total Loss만 보면 L1 Loss의 세부 개선 추이를 놓치기 쉬움.

### 3.4 ACT 학습 실행 템플릿 (기본 권장)
```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="outputs/train/${RUN_NAME}" \
  --job_name="${RUN_NAME}" \
  --batch_size=16 \
  --steps=150000 \
  --policy.optimizer_lr=1e-4 \
  --save_freq=10000 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_USER}/${RUN_NAME}"
```

---

## 4. Training Family B: SmolVLA (Vision-Language-Action)

자연어 지시문 기반 멀티태스크 및 시각-언어-행동 파운데이션 모델 파인튜닝.

### 4.1 Full Fine-Tuning 공식 권장 하이퍼파라미터 (기본값)
비전 인코더를 동결 해제하여 현장 조명 및 근접 블록 시야를 완벽하게 학습하는 최고 성능 세팅입니다:
- **Base Model**: `lerobot/smolvla_base`
- **PEFT / LoRA**: `--policy.use_peft=false` (LoRA 미사용, 전 계층 직접 업데이트)
- **동결 해제 (Unfreeze)**:
  - `--policy.freeze_vision_encoder=false` (비전 인코더 동결 완전 해제)
  - `--policy.train_expert_only=false` (VLM 백본 전체 학습)
  - `--policy.train_state_proj=true` (관절 프로젝션 레이어 학습)
- **Steps**: **150,000 steps** (30만~45만 프레임 기준 약 5.2~5.5 Epochs 달성)
- **Batch Size**: 16 (RTX 3090 24GB 기준 VRAM 최적화)
- **Optimizer & LR**:
  - AdamW, Peak LR **`2e-5 (0.00002)`**, $\beta=(0.9, 0.95)$, Weight Decay `0.01`, Grad Clip Norm `10.0`
- **Scheduler**:
  - Cosine Decay with Warmup: Warmup `3000` steps, Decay `150000` steps, Decay LR `1e-6`
- **체크포인트 저장 주기**: `save_freq=15000` (1.5만 스텝마다 저장, 총 10개 체크포인트)
- **이미지 증강**: `--dataset.image_transforms.enable=true` (기본 활성화)

*(참고: 빠른 실험용 LoRA 학습 시에는 `--policy.use_peft=true --peft.r=64 --peft.lora_alpha=64 --policy.freeze_vision_encoder=true --policy.optimizer_lr=3e-4 --steps=40000` 사용)*

### 4.2 SmolVLA Loss 지표 해석 가이드 (👉 최종 `loss` 집중 모니터링)
SmolVLA는 Flow Matching (확률 흐름 생성) 모델이므로 L1/KLD 대신 **3단계 패딩 필터링 후 최종 `loss`**를 계산합니다:

1. **`losses_after_forward` (1단계 원시 오차)**: 순전파 직후 패딩이 포함된 전체 텐서의 Flow Matching MSE 오차.
2. **`losses_after_in_ep_bound` (2단계 경계 마스킹)**: 에피소드 끝부분을 벗어난 가짜 패딩 액션을 0으로 마스킹한 오차.
3. **`losses_after_rm_padding` (3단계 차원 패딩 제거)**: 최대 32개 모터 차원 중 SO-101의 미사용 모터 축(6축 초과분)을 제거한 유효 오차.
4. **`loss` (최종 핵심 모니터링 지표)**:
   - 모든 패딩을 제거하고 실제 6축 모터의 유효 타임스텝에 대해 계산된 **진짜 Flow Matching MSE 손실**입니다.
   - W&B에서 **`train/loss`** 및 **`val/loss`**가 부드럽게 우하향하는지 집중 모니터링합니다.

### 4.3 SmolVLA Full Fine-Tuning 학습 실행 템플릿 (기본 권장)
```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --policy.type=smolvla \
  --policy.device=cuda \
  --output_dir="outputs/train/${RUN_NAME}" \
  --job_name="${RUN_NAME}" \
  --batch_size=16 \
  --steps=150000 \
  --save_freq=15000 \
  --policy.use_peft=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=2e-5 \
  --policy.scheduler_warmup_steps=3000 \
  --policy.scheduler_decay_steps=150000 \
  --policy.scheduler_decay_lr=1e-6 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_USER}/${RUN_NAME}"
```

---

## 5. Resume vs Pretrained Continuation (이어서 학습하는 2가지 전략)

### Strategy A: 로컬 체크포인트 완전 복원 (`--resume=true`)
로컬 머신에 이전 학습 체크포인트 폴더가 온전히 보존되어 있을 때 사용:

```bash
python -m lerobot.scripts.lerobot_train \
  --config_path="outputs/train/${PREVIOUS_RUN_NAME}/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true \
  --steps=150000 \
  --save_freq=10000 \
  --wandb.enable=true
```
- **주의 (Known Error)**: LeRobot 0.5.2에서 `--resume=true` 사용 시 반드시 `--config_path`에 이전 체크포인트의 `train_config.json` 전체 경로를 지정해야 함 (누락 시 `ValueError: A config_path is expected when resuming a run` 발생).
- **특징**: 옵티마이저 모멘텀, LR 스케줄러 상태, 데이터로더 샘플러 순서까지 100% 복원되어 이전 스텝 번호부터 이어서 진행됨.

### Strategy B: Hub 가중치 기반 이어서 학습 (`--policy.pretrained_path`) — 권장
HuggingFace Hub에 업로드된 체크포인트 가중치를 로드하여 새 세션으로 추가 Epoch/Step을 학습할 때 사용:

```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
  --policy.type=act \
  --policy.pretrained_path="${HF_USER}/${BASE_MODEL_NAME}" \
  --policy.repo_id="${HF_USER}/${NEW_RUN_NAME}" \
  --output_dir="outputs/train/${NEW_RUN_NAME}" \
  --job_name="${NEW_RUN_NAME}" \
  --batch_size=16 \
  --steps=100000 \
  --save_freq=10000 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --policy.push_to_hub=true
```
- **주의 (Known Error)**: 베이스 모델 설정에 `push_to_hub: true`가 포함되어 있을 경우, 새 허브 ID(`--policy.repo_id`)를 지정하거나 `--policy.push_to_hub=false`를 명시해야 함 (누락 시 `ValueError: 'repo_id' argument missing` 발생).
- **특징**: 이전 5만 스텝 가중치가 즉시 로드되어 초반 Loss가 낮은 상태(예: L1 Loss 0.06~0.08)에서 바로 시작되며, 추가 10만 스텝 완료 시 실질적으로 총 15만 스텝의 누적 효과를 가짐.

---

## 6. Validate Checkpoint Before Robot Inference

학습 완료 후 로봇 연결 전 체크포인트 무결성 점검:
1. `config.json` 및 `model.safetensors` 가중치 정상 저장 확인.
2. PEFT/LoRA 사용 시 `adapter_config.json` 및 어댑터 가중치 포함 여부 확인.
3. 전처리기(`preprocessor.json`)에 `camera1, camera2` Rename 맵 및 정규화 통계(`stats.json`) 포함 확인.
4. GPU에서 합성 관측값(Dummy Observation)을 넣어 1회 순전파(Forward pass) 에러 유무 확인.
5. 검증 완료 후 `dataset-model-registry.md`에 결과 등록.

