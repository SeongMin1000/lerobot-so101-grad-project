---
name: so101-smolvla-rwfm-pcgrad
description: SO-101 로봇 조작 과제에서 SmolVLA 정책 모델을 725ep 통합 데이터셋 기반 Expert-Only Reward-Weighted Flow Matching(RWFM)과 Same-Color Paired PCGrad(태스크1/태스크2 배치 투영)를 단일 옵티마이저 루프로 동시 파인튜닝하고 검증·배포한다. Use for RWFM, Same-Color PCGrad, combined trainer, failure masking, gradient conflict, multi-rate cadence, or general 725ep training.
---

# SO-101 SmolVLA RWFM + Same-Color PCGrad Combined Training

이 문서는 Hugging Face LeRobot 기반 SO-101 로봇 과제에서 **SmolVLA (Flow Matching VLA)** 모델에 **Reward-Weighted Flow Matching (RWFM)**과 **Same-Color Paired PCGrad (Projecting Conflicting Gradients)**를 하나의 단일 Optimizer 루프 안에서 통합 실행하는 최신 표준 아키텍처 및 운용 지침을 정의합니다.

---

## 1. 배경 및 핵심 이론 (Why Combined Training?)

### 1.1 해결하고자 하는 이중 과제
1. **자율 복구 및 파지 정밀도 (General RWFM)**:
   - 1~2cm 미세 파지 오차 및 OOD 상황에서 로봇이 스스로 복구(Self-correction)하거나 사람 개입(HIL) 궤적을 강하게 학습해야 함.
   - 실패 궤적(Failure)은 모델이 모방하지 않도록 액션 손실에서 완전히 마스킹(`learn=False`)되어야 함.
2. **배치/탑쌓기 태스크 간 그래디언트 상충 (Same-Color PCGrad)**:
   - Task 1(색상 상자 배치)과 Task 2(색상 블록 탑 쌓기)는 동일한 5개 색상 블록을 다루지만, 블록을 쥐고 난 이후의 목표 위치와 관절 궤적이 정반대이거나 충돌함.
   - 두 태스크를 단순 혼합 학습하면 Red, Green, Wood 등 특정 색상에서 그래디언트 음수 내적($g_{T1} \cdot g_{T2} < 0$, 코사인 최대 -0.30)이 발생하여 서로의 성능을 갉아먹음.

### 1.2 핵심 결합 알고리즘
1. **Multi-Rate Cadence**:
   - 매 스텝(Every Step): 725-episode General RWFM 배치 (Clean 24 + Rollout 8, 매 3스텝마다 Clean 22 + Rollout 8 + HIL 2) 역전파 수행 $\to g_{\text{general}}$.
   - 매 6스텝(Every 6th Step): 동일 색상 블록 배치에 대해 Task 1(16)과 Task 2(16)를 micro-batch 8로 분할 역전파한 뒤, 충돌 시 상호 투영 $\to g_{\text{place}} = 0.5(\tilde{g}_1 + \tilde{g}_2)$.
2. **단일 최적화 갱신 (Strict Single Optimizer Step)**:
   - $g_{\text{final}} = \alpha g_{\text{general}} + \beta g_{\text{place}}$ (스텝당 정확히 1회의 `optimizer.step()` 및 `scheduler.step()`만 실행).
3. **Action Expert-Only 동결 계약**:
   - VLM Backbone(SmolVLM2-500M), Vision Encoder(SigLIP), State Projection을 모두 Freeze하고, **Gemma 기반 Action Expert 153개 텐서(99.85M 파라미터)**만 학습하여 일반화 언어·시각 표상을 완벽 보존.
4. **Base Normalizer 통계 잠금**:
   - 신규 데이터셋 통계가 아닌 원본 865 베이스 모델의 `policy_preprocessor_step_5_normalizer_processor.safetensors`를 고정 로드하여 좌표계 왜곡 방지.

---

## 2. 데이터셋 및 보상 규약 (Authoritative Datasets)

### 2.1 725-Episode General RWFM Dataset (`eslab1234/smolvla_rwfm_general_725ep_v1`)
* **0 ~ 574 (575 episodes)**: Clean Demonstration (정상 시연, $r = 0.0$, `learn=True`)
* **575 ~ 674 (100 episodes)**: RWFM Rollout 100ep (자율 추론 + 헛손질 + 자체 복구):
  * 정상 접근(Normal): $r = 0.0$, `learn=True`
  * 자체 복구(Self-Correction): $r = +0.4$, `learn=True`
  * 헛손질/실패(Failure): $r = -1.0$, `learn=False` (미래 50액션 손실 마스킹)
* **675 ~ 724 (50 episodes)**: Human Correction HIL 50ep ($r = +0.6$, `learn=True`)

### 2.2 Placement PCGrad Branch Datasets
* **Task 1 Branch**: `eslab1234/smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged` (120ep, Red/Green/Blue/Yellow/Wood 균등)
* **Task 2 Branch**: `eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged` (127ep, 5색 균등)
* **규약**: PCGrad 브랜치 데이터셋에는 RWFM 보상 가중치를 적용하지 않음 (표준 Flow Matching MSE 손실 유지).

---

## 3. 소스 코드 구조 및 핵심 모듈

```
project/scripts/
├── train/
│   ├── train_smolvla_rwfm_pcgrad_combined.py   # 통합 트레이너 핵심 파이썬 소스
│   └── train_smolvla_same_color_pcgrad.py       # Same-Color Sampler 및 PCGrad 투영 알고리즘
├── gpu/
│   └── train_smolvla_865_rwfm_pcgrad_combined.sh # 30k 공식 학습 실행 셸 스크립트
└── tools/
    ├── build_rwfm_general_725ep.py              # 725ep 메타데이터 및 에피소드 병합 빌더
    ├── merge_and_trim_rwfm_rollouts.py          # 100ep 롤아웃 트림 및 failure 라벨링
    └── analyze_smolvla_gradient_conflict.py     # 태스크 간 코사인/내적 충돌 분석기
```

### 3.1 주요 클래스 및 함수
* **`GeneralSourceBalancedSampler`**: Clean(575), Rollout(100), HIL(50) 덱을 분리 관리하여 에피소드 단위 균등 순환 및 Cadence 주입 관리.
* **`GeneralRWFMManager`**: `meta/rwfm_rollout_annotations.json`에서 failure 구간을 파싱하여 미래 50액션 손실을 마스킹하고, 유효 프레임에 대해 $w_i \propto \exp((R_i - \max R)/T)$ 가중치 부여.
* **`compute_pcgrad(g1, g2)`**: 두 그래디언트의 내적이 음수일 때 직교 투영($g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j$)을 적용하여 파괴적 간섭 해소.
* **Resume Determinism**: `--resume true` 재개 시 고정 난수 시드(`seed=42`)를 바탕으로 이전 스텝 수만큼 샘플러 RNG 큐를 fast-forward하여 끊김 없는 결정론적 순환 복원.

---

## 4. 학습 실행 가이드라인

### 4.1 표준 30,000 스텝 본학습 (백그라운드 실행 권장)
```bash
cd /home/eslab/lerobot
nohup bash project/scripts/gpu/train_smolvla_865_rwfm_pcgrad_combined.sh > train_30k.log 2>&1 &
```

### 4.2 중단된 학습 이어서 재개 (`RESUME=true`)
```bash
RUN_NAME="smolvla_865base_rwfm725_samecolor_pcgrad_expertonly_30k" \
RESUME=true \
bash project/scripts/gpu/train_smolvla_865_rwfm_pcgrad_combined.sh
```

---

## 5. 핵심 하이퍼파라미터 및 하드웨어 표준

* **GPU 환경**: NVIDIA GeForce RTX 3090 (24GB VRAM)
* **VRAM 실측**: Peak Allocated **8.81 GB** / Peak Reserved **9.04 GB** (약 15GB 여유 확보로 OOM 제로)
* **학습 속도**:
  * 일반 RWFM 스텝: ~1.78초
  * PCGrad 스텝: ~3.65초
  * 30,000 스텝 총 소요 시간: 약 **17.4시간**
* **Learning Rate**: $1\text{e-}6 \to 3\text{e-}7$ (Cosine Decay, Warmup 300 steps)
* **Gradient Norm Clipping**: `10.0`
* **Weights & Biases**: 실시간 그래디언트 충돌 지표(`grad/general_place_dot`, `grad/general_place_cosine`, `pcgrad/raw_cosine`) 모니터링
