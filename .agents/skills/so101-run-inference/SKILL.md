---
name: so101-run-inference
description: SO-101 ACT 및 SmolVLA 비동기 추론의 GPU policy server와 robot client를 안전하게 시작·재시작·종료하고 모델, task, 2-카메라 매핑, chunk, clamp, watchdog 설정을 검증한다. Use for inference, rollout, policy server, robot client, server address, chunk settings, camera key mode, or startup commands.
---

# SO-101 Run Inference

## 1. Load Context & Architecture

Read before acting:

1. `project/docs/agent-context/project-contract.md`
2. `project/docs/agent-context/hardware-and-paths.md`
3. `project/docs/agent-context/decisions-and-known-failures.md`
4. `project/config/experiment-profiles/active.env`

### 역할 분담 (GPU Server vs Robot Client)
* **GPU PC (정책 서버)**: `policy_server`를 구동하여 GPU VRAM에 모델을 로드하고 비동기 gRPC 텐서 추론만 수행 (카메라/로봇 시리얼 장치 열지 않음).
* **Robot PC (로봇 클라이언트)**: Top/Wrist 2개 카메라 캡처, 관절 I/O, 동기화 안전 리미터(Coordinated Limiter), 모터 트래킹 와치독을 실행하며 GPU 서버와 통신.

---

## 2. Read-Only Preflight & Safety Contract

추론 시작 전 반드시 양쪽 머신에서 읽기 전용 점검을 수행합니다:

```bash
cd "${LEROBOT_ROOT:-$HOME/lerobot}"
conda activate lerobot
source project/config/experiment-profiles/active.env
python project/scripts/tools/check_antigravity_setup.py
```

### 필수 사전 점검 체크리스트
1. **2-Camera Key Contract**:
   - 체크포인트 입력 피처에 맞춰 `top -> camera1`, `wrist -> camera2` 매핑 확인.
2. **GPU 서버 연결성 (Port 8080)**:
   - 클라이언트에서 `SERVER_ADDRESS` (예: `100.85.69.64:8080`) 소켓 연결 가능 여부 확인.
3. **물리적 안전 수칙 (Safety Contract)**:
   - 작업대 5개 블록 배치 및 장애물 간섭 확인.
   - 비상정지 스위치 및 키보드 `Ctrl+C` 준비.
   - 시작 전 작업자 확인 프롬프트(`Type START`) 유지 (`SKIP_CONFIRM=false`).
   - 비정상 종료 시 로봇 낙하 방지를 위해 토크 유지 (`DISABLE_TORQUE_ON_DISCONNECT=false`).

---

## 3. Start the GPU Policy Server

GPU PC에서 정책 서버를 먼저 백그라운드 또는 전용 터미널에서 실행합니다:

```bash
cd "$LEROBOT_ROOT"
conda activate lerobot
source project/config/experiment-profiles/active.env
bash project/scripts/gpu/run_smolvla_red_policy_server.sh
```

* 서버 로그에 바인드 주소(`0.0.0.0:8080`) 및 모듈 정상 로드 메시지가 출력되는지 확인합니다.
* 클라이언트가 접속하여 모델 로드를 요청하면 GPU VRAM에 모델이 즉시 로드됩니다.

---

## 4. Start the Robot Client (Unified Async Inference Launcher)

GPU 서버가 준비되고 작업자가 물리적 시작을 승인하면 통합 런처 [`run_async_inference.sh`](file:///home/eslab/lerobot/project/scripts/robot/run_async_inference.sh)를 구동합니다:

### 4.1 기본 프로필 실행 (`active.env` 모델 자동 로드)
```bash
bash project/scripts/robot/run_async_inference.sh
```

### 4.2 ACT 모델 실행 (최적 파라미터 자동 바인딩)
```bash
POLICY_TYPE=act \
MODEL_PATH="${HF_USER}/${ACT_MODEL_NAME}" \
bash project/scripts/robot/run_async_inference.sh
```
- ACT 선택 시 `ACTIONS_PER_CHUNK=30`, `MAX_RELATIVE_TARGET=1.0°`가 자동 적용됩니다.

### 4.3 SmolVLA 모델 실행 (최적 파라미터 자동 바인딩)
```bash
POLICY_TYPE=smolvla \
MODEL_PATH="${HF_USER}/${SMOLVLA_MODEL_NAME}" \
bash project/scripts/robot/run_async_inference.sh
```
- SmolVLA 선택 시 `ACTIONS_PER_CHUNK=20`, `MAX_RELATIVE_TARGET=1.25°`가 자동 적용됩니다.

### 4.4 범용 Python CLI 직접 실행 템플릿 (참고용)
```bash
python -m lerobot.async_inference.robot_client \
  --server_address="${SERVER_ADDRESS:-100.85.69.64:8080}" \
  --policy_type="act" \
  --pretrained_name_or_path="${MODEL_PATH}" \
  --task="Pick and place 5 blocks in sequence (red, yellow, wood, green, blue)." \
  --policy_device=cuda \
  --client_device=cpu \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=follower \
  --robot.disable_torque_on_disconnect=false \
  --robot.max_relative_target=1.0 \
  --robot.max_tracking_error=35.0 \
  --robot.tracking_error_grace_steps=10 \
  --robot.cameras="{ camera1: {type: opencv, index_or_path: '$TOP_CAM', width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, camera2: {type: opencv, index_or_path: '$WRIST_CAM', width: 640, height: 480, fps: 30, fourcc: 'MJPG'} }" \
  --fps=30 \
  --actions_per_chunk=30 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name="latest_only"
```

---

## 5. Control Tuning & Safety Watchdog Rules

### 5.1 청크 및 큐 설정 (Chunk & Queue)
- **`ACTIONS_PER_CHUNK`**:
  - **ACT**: `30` (100개 예측 액션 중 앞선 30스텝(1초) 실행 $\to$ 반응성과 궤적 연속성 최적 균형).
  - **SmolVLA**: `20` ~ `30` (50개 예측 액션 중 20스텝 실행).
- **`CHUNK_SIZE_THRESHOLD`**: `0.5` (클라이언트 액션 큐가 50% 남았을 때 서버에 새 청크 비동기 선제 요청 $\to$ 딜레이 없는 매끄러운 동작).
- **`AGGREGATE_FN_NAME`**: `latest_only` (최신 관측에 기반한 최신 예측 청크 우선 적용).

### 5.2 안전 클램프 (Coordinated Limiter)
- **`MAX_RELATIVE_TARGET`**: `1.0°` ~ `1.25°` (단일 스텝당 최대 이동 각도).
- 특정 단일 관절에 급격한 델타가 요청되어도 **전체 6개 관절 벡터를 동일 비율(`scale`)로 동기화 감속**시켜 원래 의도한 방향과 타이밍을 100% 보존.

### 5.3 🔥 모터 트래킹 와치독 (그리퍼 파지 오류 방지 필수 설정)
- **문제 현상**: 물리적 블록 두께(약 2cm)로 인해 블록을 정상 파지하면 그리퍼 조(Jaw)가 약 **`23° ~ 25°`**에서 멈춤. 반면 정책 모델은 완전 닫힘(`0° ~ 2°`)을 명령하여 **약 22°의 정상적인 물리적 트래킹 오차**가 발생함.
- 만약 `MAX_TRACKING_ERROR=20.0°` (기본 5스텝)으로 낮게 잡혀 있으면, 블록을 정상적으로 집는 순간 모터 스톨(Stall)로 오인하여 **안전 중단(False Stall Abort)**이 발생함.
- **필수 설정값**:
  - **`MAX_TRACKING_ERROR=35.0`** (블록 두께 오차 허용)
  - **`TRACKING_ERROR_GRACE_STEPS=10`** (10스텝 연속 초과 시에만 실제 충돌로 판별하여 비상 정지)

---

## 6. Server State Reset & Client Reconnection

서버의 `Ready()` 핸들러는 클라이언트가 재접속할 때 다음 상태를 자동으로 초기화합니다:
- 관측 큐(`observation queue`)
- 예측 타임스텝 및 `last_processed_obs`
- 셧다운 플래그 및 디버그 캡처 ID

따라서 클라이언트 재시작 시 깨끗한 첫 번째 관측값부터 다시 시작됩니다.  
*(단, 서버 측 파이썬 코드, 전처리기, 모델 가중치가 수정된 경우에는 서버 프로세스를 완전히 재시작해야 변경사항이 반영됩니다.)*

---

## 7. Real-Time Monitoring & Emergency Stop

추론 롤아웃 중 실시간 감시 항목:
1. **카메라 프레임 드롭/지연**: USB 대역폭 및 FPS 유지 여부.
2. **모터 트래킹 오차 경고**: 그리퍼 이외의 관절(어깨, 팔꿈치)에서 지속적인 오차 발생 여부.
3. **충돌 위험 발생 시**: 즉시 `Ctrl+C` 또는 물리적 비상정지 스위치 작동.
4. **롤아웃 종료 후**: 모터 트레이스(`motor_traces`) 및 디버그 프레임을 저장하여 `so101-diagnose-runtime` 진단에 활용.

