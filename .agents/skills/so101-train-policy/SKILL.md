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

### 2.1 [필수] 이미지 증강 (Image Augmentation) 규칙: 색상 증강만 허용, 위치 왜곡(RandomAffine) 절대 금지
로봇 모방 학습에서 이미지 증강은 **반드시 물체의 물리적 좌표를 왜곡하지 않는 범위** 내에서만 적용해야 합니다:

1. 🚨 **위치 왜곡(RandomAffine) 절대 금지 (Loss 0.05 정체 & 1~2cm 빗겨남의 주원인)**:
   - LeRobot의 이미지 증강 파이프라인은 **카메라 영상만 왜곡하고 모터 액션 라벨(`action`)은 보정하지 않습니다**.
   - `RandomAffine`(회전 ±5°, 이동 ±5%)을 켜면 480×640 영상에서 2cm 블록이 약 24~32px(실물 1~2cm) 이동하지만 액션은 원래 좌표를 가리켜 **인위적인 공간 라벨 노이즈(Aleatoric Noise)**가 주입됩니다.
   - **관측된 결과**: 444ep 모델을 25만 스텝까지 학습해도 Loss가 0.05 아래로 떨어지지 않고, 실물 로봇에서 블록을 딱 1~2cm 빗겨나가 헛손질(Grasp Miss)하게 됩니다. 반면 증강이 꺼진 330ep 모델은 Loss 0.03 달성 및 정밀 파지에 성공했습니다.
2. **공식 권장 플래그**:
   - **기본 권장 (가장 안전)**: 이미지 증강 완전 비활성화 (`--dataset.image_transforms.enable=false`)
   - **조명 변화 대응 필요 시 (색상 전용 증강)**: 
     [`src/lerobot/transforms/transforms.py`](file:///home/eslab/lerobot/src/lerobot/transforms/transforms.py) 기본값에서 위치 왜곡(`affine`)이 이미 비활성화(주석 처리)되어 있으므로, 아래 옵션을 주면 **자동으로 안전한 색상 5종(`brightness`, `contrast`, `saturation`, `hue`, `sharpness`)만 활성화**됩니다:
     `--dataset.image_transforms.enable=true --dataset.image_transforms.max_num_transforms=3`

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
- **Action Chunk (고정)**: **`--policy.chunk_size=50 --policy.n_action_steps=50`** (50스텝으로 통일 고정)
- **체크포인트 저장 주기**: `save_freq=10000` (1만 스텝마다 저장)
- **이미지 증강**: `--dataset.image_transforms.enable=false` (위치 왜곡 방지를 위해 기본 OFF 권장)

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
  --dataset.image_transforms.enable=false \
  --policy.type=act \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
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

### 4.1 Full Fine-Tuning 표준 고정 파라미터 vs 가변 파라미터

> 🚨 **SmolVLA 표준 원칙**: 아래 목록에서 **[가변 파라미터]**를 제외한 모든 파라미터는 RTX 3090/4090 24GB VRAM 환경에서 OOM 없이 배치 16을 안정 구동하도록 **완전 고정**합니다. 임의로 변경하지 마십시오.

#### 🔒 1. 불변 고정 파라미터 (Fixed Standard Parameters — 수정 금지)
| 항목 | 고정 설정값 | 설정 사유 및 효과 |
| :--- | :--- | :--- |
| **환경변수 1** | `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | PyTorch 메모리 단편화 방지 |
| **환경변수 2** | `export ACCELERATE_MIXED_PRECISION="bf16"` | **배치 16 VRAM OOM 방지 핵심**: VRAM 23.5GB $\to$ 14GB (10GB 여유 확보), SmolVLM2 네이티브 포맷 무손실 1.5배 가속 |
| **비디오 백엔드** | `--dataset.video_backend="torchcodec"` | 고속 프레임 시크 및 CPU 디코딩 최적화 |
| **프레임 포맷** | `--dataset.return_uint8=true` | DataLoader IPC 메모리 절약 |
| **정규화 통계** | `--dataset.use_imagenet_stats=true` | SigLIP 비전 백본 표준 입력 스케일 |
| **이미지 증강** | `--dataset.image_transforms.enable=true`<br>`--dataset.image_transforms.max_num_transforms=3` | 기하 왜곡(RandomAffine) 없이 조명/색상 5종만 안전 적용 |
| **정책 타입** | `--policy.type=smolvla` | SmolVLA 파운데이션 모델 |
| **액션 청크** | `--policy.chunk_size=50`<br>`--policy.n_action_steps=50` | SO-101 검증 표준 50스텝 고정 |
| **실행 장치** | `--policy.device=cuda` | GPU 가속 |
| **전체 동결 해제** | `--policy.use_peft=false`<br>`--policy.freeze_vision_encoder=false`<br>`--policy.train_expert_only=false`<br>`--policy.train_state_proj=true` | LoRA 배제, VLM 전 계층 및 관절 프로젝션 Full Fine-Tuning |
| **KV 캐시 인자** | **`--policy.use_cache=false` 절대 추가 금지**<br>(기본값 `true` 유지) | **🚨 치명적 에러 유발 방지**: 학습 시엔 내부 코드에서 `use_cache=False`가 하드코딩되어 있어 CLI에 이 옵션을 줘도 VRAM이 1바이트도 줄지 않음. 반면 이 옵션을 주면 체크포인트 `config.json`에 `false`로 저장되어 추론(`sample_actions`) 시 Action Expert KV 캐시가 비어 **`The size of tensor a (227) must match the size of tensor b (50)`** 런타임 크래시 발생 |
| **배치 사이즈** | `--batch_size=16` | RTX 3090/4090 24GB 기준 표준 배치 |
| **데이터로더** | `--num_workers=8`<br>`--prefetch_factor=2`<br>`--persistent_workers=true` | 8개 워커와 큐 버퍼 2배율로 VRAM/RAM 오버헤드 최소화 |
| **로깅 및 허브** | `--wandb.enable=true`<br>`--wandb.project="lerobot-so101-grad"`<br>`--policy.push_to_hub=true` | 학습 지표 모니터링 및 HF Hub 자동 백업 |

#### 🎛️ 2. 사용자 조절 가변 파라미터 (User Configurable Parameters — 상황에 맞게 입력)
- `--dataset.repo_id`: 학습 데이터셋 Hugging Face Hub ID (예: `eslab1234/multitask_5blocks_v3_704ep_hil_r1_merged`)
- `--policy.pretrained_path`: 시작 체크포인트 경로 (예: `outputs/train/smolvla_multitask_5blocks_v3_575ep_fullft_b16_300k/checkpoints/285000/pretrained_model` 또는 `lerobot/smolvla_base`)
- `--steps`: 총 학습 스텝 수 (예: `40000` 또는 `150000`)
- `--policy.scheduler_decay_steps`: 감쇄 스텝 수 (**반드시 `--steps`와 1:1로 동일하게 일치**)
- `--policy.optimizer_lr`: 학습률 (이어서 미세 조정 시 `1e-5`, 신규 데이터 학습 시 `2e-5`)
- `--policy.scheduler_warmup_steps`: 웜업 스텝 (미세 조정 시 `1500`, 대규모 세션 시 `3000`)
- `--policy.scheduler_decay_lr`: 최저 학습률 (`1e-6`)
- `--save_freq`: 체크포인트 저장 주기 (예: `5000` 또는 `15000`)
- `--output_dir` / `--job_name`: 출력 디렉터리 및 작업명
- `--policy.repo_id`: HF Hub 모델 업로드 경로 (예: `eslab1234/${RUN_NAME}`)

---

### 4.2 💡 VRAM 24GB 환경에서 배치 16 OOM 발생 원인 및 해결 (트러블슈팅)

1. **왜 이전 575ep 30만 스텝 때는 안 터졌는데, 지금 30MB 부족으로 터졌는가?**:
   - FP32(단정밀도) 모드에서 SmolVLA 배치 16은 모델(1.8GB) + 옵티마이저(4.8GB) + 16개 트랜스포머 레이어의 Activation(16.8GB)으로 인해 **기본적으로 VRAM 23.4~23.5GB(99.8%)를 소모하며 칼날 위에서 턱걸이**를 하고 있었습니다.
   - 에러 로그: `Tried to allocate 30.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 18.88 MiB is free.`
   - 704ep 데이터셋에 긴 자연어 태스크 프롬프트가 병합되면서 토큰 시퀀스 길이가 단 2~3개 늘어났고, 16개 어텐션 레이어를 통과하며 **정확히 30MB가 초과**되어 OOM이 발생한 것입니다.
2. **해결책**:
   - `export ACCELERATE_MIXED_PRECISION="bf16"`을 실행 세션에 1줄 선언해주면 어텐션 텐서가 2바이트로 계산되어 **VRAM이 23.5GB에서 14GB 내외로 즉시 10GB 가까이 여유가 생기며 절대 OOM이 나지 않습니다**.
   - SmolVLM2 원본 백본이 bfloat16이므로 정밀도 손실 없이 학습 속도도 1.5배 빨라집니다.

---

### 4.3 SmolVLA Loss 지표 해석 가이드 (👉 최종 `loss` 집중 모니터링)
SmolVLA는 Flow Matching (확률 흐름 생성) 모델이므로 L1/KLD 대신 **3단계 패딩 필터링 후 최종 `loss`**를 계산합니다:

1. **`losses_after_forward` (1단계 원시 오차)**: 순전파 직후 패딩이 포함된 전체 텐서의 Flow Matching MSE 오차.
2. **`losses_after_in_ep_bound` (2단계 경계 마스킹)**: 에피소드 끝부분을 벗어난 가짜 패딩 액션을 0으로 마스킹한 오차.
3. **`losses_after_rm_padding` (3단계 차원 패딩 제거)**: 최대 32개 모터 차원 중 SO-101의 미사용 모터 축(6축 초과분)을 제거한 유효 오차.
4. **`loss` (최종 핵심 모니터링 지표)**:
   - 모든 패딩을 제거하고 실제 6축 모터의 유효 타임스텝에 대해 계산된 **진짜 Flow Matching MSE 손실**입니다.
   - **Loss 기준치 해석**:
     - `0.03 대`: 정상 수렴. 픽셀-모터 좌표가 1:1로 일치하여 2cm 블록의 밀리미터 단위 정밀 파지가 가능함.
     - `0.05 대 (정체)`: RandomAffine 위치 왜곡(라벨 노이즈) 주입, 청크 60스텝 확장, 또는 멀티태스크 과소적합의 전형적 증상. 실물 로봇에서 1~2cm 빗겨나는 헛손질 발생.

---

### 4.4 SmolVLA Full Fine-Tuning 학습 실행 템플릿 (공식 표준)

```bash
# 1. 필수 환경변수 선언 (메모리 단편화 방지 및 VRAM 14GB 안정화)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_MIXED_PRECISION="bf16"

# 2. 고정 세팅 기반 학습 실행 (가변 파라미터만 조정)
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
  --dataset.video_backend="torchcodec" \
  --dataset.return_uint8=true \
  --dataset.use_imagenet_stats=true \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --policy.type=smolvla \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.device=cuda \
  --policy.use_peft=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=2e-5 \
  --policy.scheduler_warmup_steps=3000 \
  --policy.scheduler_decay_steps=150000 \
  --policy.scheduler_decay_lr=1e-6 \
  --batch_size=16 \
  --num_workers=8 \
  --prefetch_factor=2 \
  --persistent_workers=true \
  --steps=150000 \
  --save_freq=15000 \
  --output_dir="outputs/train/${RUN_NAME}" \
  --job_name="${RUN_NAME}" \
  --wandb.enable=true \
  --wandb.project="lerobot-so101-grad" \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_USER}/${RUN_NAME}"
```

---

## 5. Resume vs Pretrained Continuation (이어서 학습하는 2가지 전략)

### Strategy A: 로컬 체크포인트 완전 복원 (`--resume=true`)
로컬 머신에서 중단된 학습을 동일한 조건으로 끝까지 재개할 때 사용:

```bash
python -m lerobot.scripts.lerobot_train \
  --config_path="outputs/train/${PREVIOUS_RUN_NAME}/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true \
  --steps=150000 \
  --save_freq=10000 \
  --wandb.enable=true
```
* 🚨 **치명적 주의 (10만 스텝 유령 학습 참사 방지)**:
  - 15만 스텝으로 완료된 모델에서 스텝만 늘려 `--resume=true --steps=250000`으로 돌리면, **스케줄러 감쇄가 이미 15만 스텝에서 끝나버렸기 때문에 이후 10만 스텝(150,001~250,000) 내내 학습률이 최저치인 `1e-6` (0.000001)으로 바닥에 굳은 채 실행**됩니다.
  - AdamW에서 LR `1e-6`은 가중치가 전혀 갱신되지 않는 사실상의 "동결(Freeze)" 상태이므로, 추가 10만 스텝 동안 Loss가 단 0.001도 떨어지지 않는 원인이 됩니다.
  - 따라서 단순 학습 중단 복구가 아니라 **"체크포인트 기반 추가 학습"을 할 때는 무조건 아래 Strategy B를 사용**해야 합니다.

---

### Strategy B: Hub/로컬 가중치 기반 신규 세션 파인튜닝 (`--policy.pretrained_path`) — 🌟 강력 권장
이전 체크포인트 가중치를 시작점으로 삼아, **싱싱한 새 학습률 스케줄러(Warmup $\to$ Cosine Decay)를 부여하여 추가 Epoch를 학습**시키는 정석 방법입니다:

#### 📌 필수 하이퍼파라미터 세팅 규칙
> **핵심 원칙**: 추가 학습 세션에서는 **반드시 `--steps`와 `--policy.scheduler_decay_steps`를 1:1로 동일하게 일치**시키고, 기존 가중치 충격 방지를 위해 **짧은 웜업(1000~2000 스텝)**을 부여해야 합니다.

```bash
# 필수 환경변수 선언
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_MIXED_PRECISION="bf16"

# 체크포인트 기반 이어서 파인튜닝 (배치 16 불변 고정 표준)
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
  --dataset.video_backend="torchcodec" \
  --dataset.return_uint8=true \
  --dataset.use_imagenet_stats=true \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --policy.type=smolvla \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.device=cuda \
  --policy.pretrained_path="${PRETRAINED_MODEL_PATH}" \
  --policy.use_peft=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=1e-05 \
  --policy.scheduler_warmup_steps=1500 \
  --policy.scheduler_decay_steps=${TOTAL_STEPS} \
  --policy.scheduler_decay_lr=1e-06 \
  --batch_size=16 \
  --num_workers=8 \
  --prefetch_factor=2 \
  --persistent_workers=true \
  --steps=${TOTAL_STEPS} \
  --save_freq=5000 \
  --output_dir="outputs/train/${RUN_NAME}" \
  --job_name="${RUN_NAME}" \
  --wandb.enable=true \
  --wandb.project="lerobot-so101-grad" \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_USER}/${RUN_NAME}"
```

---

## 6. Validate Checkpoint Before Robot Inference

학습 완료 후 로봇 연결 전 체크포인트 무결성 점검:
1. `config.json` 및 `model.safetensors` 가중치 정상 저장 확인.
2. PEFT/LoRA 사용 시 `adapter_config.json` 및 어댑터 가중치 포함 여부 확인.
3. 전처리기(`preprocessor.json`)에 `camera1, camera2` Rename 맵 및 정규화 통계(`stats.json`) 포함 확인.
4. GPU에서 합성 관측값(Dummy Observation)을 넣어 1회 순전파(Forward pass) 에러 유무 확인.
5. 검증 완료 후 `dataset-model-registry.md`에 결과 등록.

---

## 7. Offline Reinforcement Learning (Reward-Weighted Flow-Matching / RWFM)

실물 로봇의 수동 리셋 피로도 없이, HIL 전체 롤아웃(정상 0.8 + 헛손질 0.1 + 사람 교정 1.0) 데이터셋에서 **50-Chunk 행동 가치에 비례하여 Flow-Matching 손실 가중치를 부여하는 오프라인 강화학습**입니다.

### 7.1 핵심 수식 및 원리
LeRobot 공식 `sample_weighter` 및 [`RewardSampleWeighter`](file:///home/eslab/lerobot/src/lerobot/utils/sample_weighting.py)를 사용하여, 각 학습 샘플의 미래 50프레임 액션 청크에 대한 평균 보상($R_i$)을 계산하고 지수 가중치를 곱합니다:

$$R_i = \frac{1}{K}\sum_{k=0}^{K-1} r_{t+k}, \quad w_i = \exp\left(\frac{R_i - \max(R)}{T}\right), \quad \mathcal{L}_{\text{RWFM}} = \frac{\sum_i w_i \cdot \mathcal{L}_{\text{Flow-MSE}}(i)}{\sum_i w_i}$$

* **사람 개입 교정/복구 구간 ($r=1.0$)**: $w \approx 1.0$ (최대 가중치로 모범답안 강력 모방).
* **정상 자율주행 접근 구간 ($r=0.8$)**: $w \approx 0.67$ (안정적인 기본 주행 학습 유지).
* **로봇 빗나감/실패 구간 ($r=0.1$)**: $w \approx 0.16$ (가중치가 대폭 축소되어 헛손질 모방 억제).

### 7.2 프레임/구간 보상 점수표 자동 생성
HIL 촬영 시 기록된 개입 타임라인(`meta/episode_interventions.json`)을 기반으로 구간별 점수 매핑 생성:
```bash
python project/scripts/tools/generate_frame_rewards.py \
  --repo_id="${HF_USER}/${DATASET_NAME}" \
  --output="project/config/episode_frame_rewards.json" \
  --normal_reward=0.8 \
  --failure_reward=0.1 \
  --correction_reward=1.0 \
  --mistake_window=45
```
*(참고: `dataset_root` 내 `meta/episode_interventions.json`이 존재하면 학습 시 별도 파일 지정 없이도 자동 감지됨)*

### 7.3 RWFM 오프라인 RL 학습 실행 템플릿
```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
  --dataset.image_transforms.enable=false \
  --policy.type=smolvla \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.device=cuda \
  --batch_size=16 \
  --steps=150000 \
  --sample_weighting.type=reward_weighted \
  --sample_weighting.frame_reward_path="project/config/episode_frame_rewards.json" \
  --sample_weighting.temperature=0.5 \
  --sample_weighting.chunk_size=50 \
  --output_dir="outputs/train/${RUN_NAME}" \
  --job_name="${RUN_NAME}" \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_USER}/${RUN_NAME}"
```
- **`--sample_weighting.temperature`**: 기본값 0.5 (작을수록 1.0점 복구 궤적에 학습 집중, 클수록 균등 BC에 근접).
- **체크포인트 호환성**: 모델 구조(Action Expert / Vision)는 전혀 변경되지 않으므로, 일반 `run_async_inference.sh` 추론 서버 및 Jetson 클라이언트에서 그대로 100% 호환 구동.


