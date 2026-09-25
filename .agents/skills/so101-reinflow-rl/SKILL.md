---
name: so101-reinflow-rl
description: SO-101 로봇 조작 과제에서 SmolVLA(Flow Matching VLA) 모델에 ReinFlow(Flow-SDE, NeurIPS 2025)를 적용하여 온라인 강화학습(Flow-PPO), 자율 리셋, 자동 보상 채점, 비동기 추론 가속(K=4)을 수행하고 파인튜닝한다. Use for ReinFlow, SmolVLA RL, Flow-PPO, online reinforcement learning, reward engineering, stochastic flow denoising, or rollout buffer.
---

# SO-101 ReinFlow Online RL (SmolVLA Fine-Tuning)

이 문서는 Hugging Face LeRobot 기반 SO-101 6-DOF 로봇 조작 과제에서 **SmolVLA (Flow Matching VLA)** 모델을 **ReinFlow (NeurIPS 2025)** 알고리즘 기반 온라인 강화학습(Flow-PPO)으로 파인튜닝하는 전체 아키텍처, 코드 구현, 롤아웃 수집 방법 및 운용 가이드라인을 정의합니다.

---

## 1. 개요 및 핵심 이론 (Why ReinFlow with SmolVLA?)

### 1.1 해결하고자 하는 문제
* 기존 모방학습(Behavior Cloning)으로 학습된 SmolVLA는 시연 데이터의 분포 외 영역(OOD)이나 1~2cm 미세 파지 오차 상황에서 스스로 궤적을 복구하지 못하는 한계가 있습니다.
* 그러나 일반적인 Flow Matching / Diffusion 모델은 결정론적 ODE 경로를 사용하므로, **행동 확률 밀도 $\log \pi(a|s)$ 계산이 불가능**하여 PPO/DPO 등 표준 강화학습 적용이 어려웠습니다.

### 1.2 ReinFlow의 핵심 원리
1. **확률적 마르코프 과정 변환 (Flow-SDE)**:
   결정론적 속도장 $v_\theta(x_t, t, s)$ 적분 경로에 단계별 가우시안 노이즈 $\sigma_k \epsilon_k$를 주입하여 이산 시간 마르코프 체인으로 변환:
   $$x_{k+1} = x_k + \Delta t \cdot v_\theta(x_k, t_k, s) + \sigma_k \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, I)$$
2. **해석적 Log-Probability 계산**:
   각 디노이징 전이의 가우시안 확률을 누적하여 액션 청크 $a$의 정확한 로그 확률을 계산:
   $$\log \pi_\theta(a|s) = \sum_{k=0}^{K-1} \log \mathcal{N}\left(x_{k+1}; x_k + \Delta t \cdot v_\theta(x_k, t_k, s), \sigma_k^2 I\right)$$
3. **초고속 저스텝 RL ($K=4$)**:
   Diffusion Policy(10~20스텝) 대비 **단 4스텝**만으로 안정적인 PPO 그래디언트 역전파를 수행하며, 비동기 추론 지연시간을 ~180ms에서 **~50ms로 70% 단축**합니다.

### 1.3 $\pi_0$ (OpenPI)와의 호환성
* SmolVLA는 Physical Intelligence의 $\pi_0$(`openpi`) 모델 구조(SmolVLM2 백본 + Gemma 기반 Action Expert)를 경량화하여 LeRobot에 이식한 동일 아키텍처입니다.
* 따라서 ReinFlow 공식 논문 및 구현체의 `Flow-SDE` 메커니즘이 SmolVLA에 1:1로 정확하게 적용됩니다.

---

## 2. 소스 코드 아키텍처 및 구현 모듈

기존 LeRobot 코드를 건드리지 않고, [`src/lerobot/grad_project/rl/`](file:///home/eslab/lerobot/src/lerobot/grad_project/rl/) 디렉터리 내에 모듈화되어 있습니다.

```
src/lerobot/grad_project/rl/
├── __init__.py                # RL 패키지 진입점
├── reinflow_smolvla.py        # SmolVLAReinFlowPolicy (Stochastic Denoising & Log-Prob, Critic Head)
├── buffer.py                  # ReinFlowRolloutBuffer (GAE Advantage 계산 및 배치 샘플러)
├── reward_evaluator.py        # AutoRewardEvaluator (OpenCV 슬롯 감지 + Gripper + Jerk 페널티)
├── trainer.py                 # ReinFlowPPOTrainer (Clipped PPO + Value + KL + Entropy 손실)
└── real_robot_rollout.py      # 실물 로봇 및 Mock 자가테스트 실행 러너
```

### 2.1 주요 클래스 역할
* **`SmolVLAReinFlowPolicy`**: `SmolVLAPolicy`를 상속하여 4-Step 확률적 액션 샘플링(`sample_actions_stochastic`)과 궤적 재평가(`evaluate_trajectory_log_prob`)를 제공.
* **`SmolVLACriticHead`**: Action Expert 히든 상태로부터 가치 함수 $V(s)$를 추정하는 2-Layer MLP.
* **`ReinFlowRolloutBuffer`**: `(obs, action, trajectory, log_prob, reward, done, value)` 튜플 저장 및 Generalized Advantage Estimation (GAE) 계산.
* **`AutoRewardEvaluator`**: 탑 카메라 영상과 OpenCV 타깃 판정기([`opencv_target_verifier.py`](file:///home/eslab/lerobot/src/lerobot/grad_project/perception/opencv_target_verifier.py))를 연동하여 자율 보상 산출.
* **`ReinFlowPPOTrainer`**: Flow-PPO Clipped Surrogate Objective와 SFT 참조 모델과의 KL 발산 제약(`kl_coeff`)을 최적화.

---

## 3. 실물 로봇 롤아웃 및 보상 설계 (Rollout & Reward Contract)

### 3.1 4단계 롤아웃 루프
1. **관측 자세 자동 복귀 (Auto-Reset)**: 에피소드 종료 시 로봇 팔이 미리 지정된 안전 관측 자세(`observe_pose`)로 자동 이동.
2. **확률적 추론 (Stochastic Sampling)**: GPU 정책 서버가 4-Step Flow-SDE로 $(s_t, a_t, \log \pi_\theta(a_t|s_t), \text{trajectory})$ 생성.
3. **액션 실행 & 보상 채점 (Execute & Auto-Reward)**: 로봇이 50Hz로 50-chunk 액션을 실행하고, Top 카메라 OpenCV가 타깃 슬롯 점유 여부 판정.
4. **PPO 정책 갱신 (Policy Update)**: 버퍼에 8~10 에피소드가 채워지면 PPO 역전파 수행 및 최신 체크포인트 자동 저장.

### 3.2 보상 함수 (Reward Formulation)
$$R_{total} = R_{step} + R_{smooth} + R_{gripper} + R_{success}$$

| 보상 요소 | 수식 / 조건 | 기본값 |
| :--- | :--- | :---: |
| **Step Penalty ($R_{step}$)** | 빠른 작업 완료 유도 | `-0.02` / step |
| **Action Smoothness ($R_{smooth}$)** | 저크/진동 방지: $-\lambda_{smooth} \|\Delta^2 a\|^2$ | `-0.01` * jerk |
| **Gripper Bonus ($R_{gripper}$)** | 블록 근접 시 그리퍼 닫힘 유지 | `+0.50` |
| **Target Success ($R_{success}$)** | OpenCV 타깃 슬롯(Red/Yellow/Blue 등) 점유 감지 시 | **`+10.0`** (Terminal) |

---

## 4. 환경변수 및 실행 파라미터 가이드

[`run_smolvla_reinflow_rl.sh`](file:///home/eslab/lerobot/project/scripts/gpu/run_smolvla_reinflow_rl.sh)에서 설정할 수 있는 전체 환경변수입니다:

| 환경변수명 | 기본값 | 설명 |
| :--- | :---: | :--- |
| **`CHECKPOINT_PATH`** | `""` | 사전학습된 SmolVLA 모델 경로 또는 Hugging Face Repo ID |
| **`TARGET_COLOR`** | `"red"` | 목표 블록 색상 (`red`, `yellow`, `green`, `blue`, `wood`) |
| **`NUM_EPISODES`** | `20` | 총 수행할 강화학습 에피소드 횟수 |
| **`STEPS_PER_EPISODE`** | `25` | 1 에피소드당 최대 액션 청크 수 |
| **`BATCH_SIZE`** | `8` | PPO 정책 업데이트를 트리거할 에피소드 버퍼 크기 |
| **`PPO_EPOCHS`** | `4` | 수집된 버퍼 데이터를 가지고 PPO 학습을 반복할 에폭 수 |
| **`ACTOR_LR`** | `3e-5` | Action Expert 학습률 (SFT 대비 작게 설정) |
| **`CRITIC_LR`** | `1e-4` | Critic 가치 신경망 학습률 |
| **`SIGMA`** | `0.05` | ReinFlow 노이즈 주입 스케일 (0.03 ~ 0.08 권장) |
| **`OUTPUT_DIR`** | `"outputs/reinflow_smolvla"` | 훈련된 모델 체크포인트 저장 폴더 |
| **`CUDA_DEVICE`** | `"0"` | 사용할 GPU 디바이스 번호 |
| **`MOCK_FLAG`** | `""` | 로봇 없이 가상 자가 테스트 시 `"--mock"`, 실제 로봇 구동 시 `""` |

---

## 5. 실행 명령어 (Runbooks)

### 5.1 가상 자가 테스트 (Mock Verification)
로봇 하드웨어 없이 알고리즘 파이프라인 무결성을 점검할 때 사용합니다:
```bash
conda activate lerobot
cd /home/eslab/lerobot

MOCK_FLAG="--mock" \
NUM_EPISODES=2 \
STEPS_PER_EPISODE=3 \
BATCH_SIZE=2 \
./project/scripts/gpu/run_smolvla_reinflow_rl.sh
```

### 5.2 유닛 테스트 실행 (PyTest)
```bash
/home/eslab/miniforge3/envs/lerobot/bin/pytest tests/grad_project/test_reinflow_smolvla.py -svv
```

### 5.3 실제 로봇 온라인 강화학습 (Real Robot Online RL)
사전학습된 체크포인트를 기반으로 실물 로봇에서 파지 정밀도 파인튜닝을 진행합니다:
```bash
conda activate lerobot
cd /home/eslab/lerobot

CHECKPOINT_PATH="eslab1234/smolvla_red_full_138ep_recovery_lora_r64_lr1e3_20k_v1" \
TARGET_COLOR="red" \
NUM_EPISODES=30 \
BATCH_SIZE=8 \
ACTOR_LR=3e-5 \
SIGMA=0.05 \
./project/scripts/gpu/run_smolvla_reinflow_rl.sh
```

---

## 6. 트러블슈팅 및 런타임 진단

1. **CPU 모드로 잡힐 때 (`Switching to 'cpu'`)**:
   * 현 머신에 PyTorch CUDA 빌드가 정상 설치되어 있는지 `python -c "import torch; print(torch.cuda.is_available())"`로 확인.
   * GPU 머신에서 실행할 경우 `CUDA_VISIBLE_DEVICES=0` 확인.
2. **모터 진동 및 급발진 발생 시**:
   * `reward_evaluator.py`의 `action_smoothness_coeff`를 `0.01` ➔ `0.03`으로 상향.
   * `trainer.py`의 `kl_coeff`를 `0.05` ➔ `0.10`으로 올려 기존 SFT 기준 궤적에서 크게 벗어나지 않도록 규제.
3. **탐색 부족으로 성공 보상이 안 들어올 때**:
   * `SIGMA` 값을 `0.05` ➔ `0.08`로 올려 초기 궤적 탐색 반경 확장.
   * 작업자 개입(Human-in-the-Loop)으로 블록을 살짝 건드려 유효 영역 내 진입 유도.
