---
name: so101-record-data
description: LeRobot SO-101 졸업과제의 5-블록 원테이크(v3) 하이브리드 시연 촬영, RBF 관절 모델 및 눈금자 보정, episode 재촬영·폐기, offline recovery 수집, 데이터셋 검사·병합 표준을 수행한다. Use for record, filming, demonstration, dataset collection, 5-block one-take v3, episode quality, or coverage questions.
---

# SO-101 Record Data

## Load context

Read before acting:

1. `project/docs/agent-context/project-contract.md`
2. `project/docs/agent-context/hardware-and-paths.md`
3. `project/docs/agent-context/decisions-and-known-failures.md`
4. `project/config/experiment-profiles/active.env`

Read `dataset-model-registry.md` when extending or merging an existing dataset.

## Primary Recording Mode: Hybrid 5-Block One-Take v3

Task 1 및 5-블록 조작의 공식 표준 수집 모드:

- **Hybrid 5-Block One-Take Recording (`v3`)** (`hybrid_record_5blocks_onetake_v3.py`):
  - 5개 블록(`red -> yellow -> wood -> green -> blue`)의 픽앤플레이스 시연을 **1개의 연속 에피소드**로 끊김 없이 녹화.
  - **자동 호버 진입 (Auto-approach)**: Observe 포즈에서 블록 상공 호버 위치까지 **45점 티칭 RBF 관절 모델 (`hover_joint_model_record.json`)**을 통해 부드러운 C2-continuous 스플라인 궤적으로 자동 비행.
  - **수동 정밀 조작 (Manual Teleop)**: 호버 도달 후 리더암 텔레옵으로 블록 파지, 타겟 슬롯 이동, 정밀 안착 및 릴리즈 수행.
  - **자동 복귀 및 전환**: 슬롯 배치 후 다음 블록 호버 위치로 자동 이동하며, 마지막 5번째 블록 안착 시 그리퍼를 닫고 `task_complete` 포즈로 자동 복귀 후 녹화 동결.
  - 런처 스크립트: `project/scripts/robot/run_hybrid_5blocks_onetake_v3_record.sh`
  - Python 진입점: `python -m lerobot.grad_project.recording.hybrid_record_5blocks_onetake_v3`

*(보조 모드: 단일 블록 엔드투엔드 수집 `smolvla_record_observe_return.py`, 실패 상태 복구 시연 `teleop_before_episode=true`)*

## Calibration & Kinematic Mapping Stack

### 1. Top Camera 34-Point Ruler Homography & Distance Calibration
- 파일: `project/config/grasp_pixel_to_robot_record.json`
- 백업 데이터: `project/config/ruler_calibration_samples_record.json`
- 탑 카메라 픽셀 $(cx, cy) \to$ 실제 테이블 좌표 $(X, Y\text{ m})$ 및 거리 $R = \sqrt{X^2 + Y^2}$, 방위각 $\theta = \text{atan2}(Y, X)$ 매핑.
- 34개 물리 눈금자 그리드 점($X \in [0, 35]\text{cm}, Y \in [-30, +30]\text{cm}$)으로 피팅되어 카메라 렌즈 왜곡 보정.
- 평균 잔차 오차: **$2.03\text{ mm}$**, 최대 오차: **$4.36\text{ mm}$**.
- 실시간 YOLO 및 좌표 검증: `python project/scripts/tools/test_yolo_live.py`

### 2. 45-Point 2-Step Teleoperation Demonstration Joint Mapping (RBF Model)
- 티칭 도구: `project/scripts/tools/teach_block_hover_joints.py`
- 수집 샘플: `project/config/hover_demonstration_samples_record.json`
- 학습 모델: `project/config/hover_joint_model_record.json`
- **2단계 티칭 워크플로우**:
  - `[Step 1: Snapshot]`: 로봇 암이 뒤로 빠진 상태에서 탑 카메라로 가림 없는 블록 픽셀 $(cx, cy) \to (X, Y)$ 캡처.
  - `[Step 2: Teach]`: 작업자가 리더암으로 팔로워를 블록 정중앙 호버 위치로 유도 후 기록 (기구학 IK의 한계를 100% 실측으로 극복).
- **RBF Multiquadric Interpolator**:
  - 9개 방향 $\times$ 5개 거리($R \in [11, 38]\text{cm}$) 총 45개 실측 포인트.
  - 전체 평균 관절 오차: **$0.22^\circ$** (`shoulder_pan`: $0.04^\circ$, `shoulder_lift`: $0.34^\circ$, `elbow_flex`: $0.41^\circ$, `wrist_flex`: $0.15^\circ$, `wrist_roll`: $0.16^\circ$).

### 3. Distance-Adaptive Wrist Camera Elevation & Earlier Pan Alignment
- **조기 Pan 정렬 (`shoulder_pan`)**: $\text{pan\_ratio} = 0.90 - 0.15 \times \text{clip}((R - 0.12) / 0.25, 0, 1)$ (비행 시간의 $75\%\sim 90\%$ 시점에 Pan 정렬 완료).
- **손목 카메라 시야 확보 (Wrist Bump)**: $\text{wrist\_bump} = 6.0^\circ + 8.0^\circ \times \text{clip}((R - 0.12) / 0.25, 0, 1)$ ($+6^\circ$ 근거리 $\to +14^\circ$ 원거리).
- `cmd["wrist_flex.pos"] = base_val - wrist_bump * sin(pi * s)` 적용으로 비행 중 손목을 살짝 들어 블록을 손목 카메라 시야 중앙에 유지.

### 4. Vertical Lift Clearance & Slot Escape (슬롯 수직 탈출 리프트)
- 슬롯에 블록 배치 후 다음 블록으로 이동 시(`return_to_observe_each_block=false`), 대각선 이동으로 인한 슬롯/블록 충돌을 방지:
  - **0.6초간 제자리 수직 상승**: 어깨 상승(`shoulder_lift.pos -= 25.0°`), 팔꿈치 리프트(`elbow_flex.pos -= 15.0°`).
  - **손목 하향 각도 고정**: `wrist_flex.pos = min(98.0, cur + 20.0°)`로 그리퍼 핑거를 바닥 쪽으로 수직 정렬하여 슬롯 벽면 간섭 방지.

### 5. Distance-Adaptive Pan Bias Compensation (우측 호버 드리프트 보정)
- 물리 하중 및 기구 유격으로 인한 원거리 우측 쏠림을 보정하기 위한 거리 기반 Pan 보정:
  - 설정: `--pan_bias_direction=right` (또는 `left`, `none`)
  - 근거리 보정: `--pan_bias_near_deg=1.0°` ($R \le 12\text{cm}$)
  - 원거리 보정: `--pan_bias_far_deg=3.5°` (또는 최대 $6.5^\circ$, $R \ge 37\text{cm}$)

### 6. Task Complete Pose & Freeze
- 5번째 블록 배치가 완료되면 그리퍼를 45°(닫힘)로 닫고 `project/config/runtime.json`의 `task_complete` 포즈로 이동.
- 완료 포즈 도달 즉시 프레임 레코딩을 프리즈하여 군더더기 없는 에피소드 종료 보장.

## Preflight

수집 시작 전 읽기 전용 점검:

```bash
cd "${LEROBOT_ROOT:-$HOME/lerobot}"
conda activate lerobot
source project/config/experiment-profiles/active.env

python -m lerobot.grad_project.recording.hybrid_record_5blocks_onetake_v3 --help
python -m lerobot.grad_project.paths
ls -l "$ROBOT_PORT" "$TELEOP_PORT" "$TOP_CAM" "$WRIST_CAM"
```

카메라 점검:
- 2개 카메라(`top`, `wrist`) 모두 **640x480 @ 30 FPS MJPG** 확인.
- 탑 카메라 V4L2 파라미터는 팩토리 기본값(Auto WB=1, Brightness=0, Contrast=40, Saturation=64, Gamma=300) 확인.
- 작업 공간 안전, 비상정지 스위치, 시작 전 Observe 포즈 정렬 확인 후 실행.

## Generate a Recording Command

최신 v3 런처 실행:

```bash
bash project/scripts/robot/run_hybrid_5blocks_onetake_v3_record.sh
```

또는 직접 Python CLI 실행:

```bash
DATASET_NAME="task1_hybrid_5blocks_v3_sessionN"
NUM_EPISODES="10"

python -m lerobot.grad_project.recording.hybrid_record_5blocks_onetake_v3 \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=follower \
  --robot.disable_torque_on_disconnect=false \
  --teleop.type=so101_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id=leader \
  --robot.cameras="{ top: {type: opencv, index_or_path: '$TOP_CAM', width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '$WRIST_CAM', width: 640, height: 480, fps: 30, fourcc: 'MJPG'} }" \
  --dataset.repo_id="$HF_USER/$DATASET_NAME" \
  --dataset.single_task="Pick and place 5 blocks in sequence (red, yellow, wood, green, blue)." \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.fps=30 \
  --dataset.video=true \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --color_sequence="red,yellow,wood,green,blue" \
  --max_blocks=5 \
  --macro_goto_duration_s=2.0 \
  --macro_return_duration_s=2.0 \
  --hover_z_offset_m=0.10 \
  --yolo_model_path="project/models/yolo_block_detector/best.pt" \
  --grasp_calibration="project/config/grasp_pixel_to_robot_record.json" \
  --runtime_config="project/config/runtime.json" \
  --pan_bias_direction="right" \
  --pan_bias_near_deg=1.0 \
  --pan_bias_far_deg=3.5 \
  --play_sounds=true
```

## Operator Contract (조작자 키보드 제어 계약)

- **`→` (오른쪽 화살표)**: 5개 블록이 정상 배치되고 완료 포즈로 복귀한 후 **에피소드 저장**.
- **`←` (왼쪽 화살표)**: 중간에 블록을 떨어뜨리거나 파지에 실패한 경우 즉시 **버퍼 폐기**, Observe로 복귀 후 **동일 에피소드 번호 재촬영**.
- **`Enter`**: 에피소드 시작 대기 상태(`wait_enter_before_episode`)에서 다음 에피소드 시작.
- **`Escape`**: 현재 에피소드를 저장하지 않고 즉시 폐기하며 세션 안전 종료.
- 자동 복귀 및 전환 구간은 데이터셋 밖에서 처리되거나 스플라인 궤적으로 부드럽게 유지됨.

## Offline Recovery Workflow

1. 추론 중 타겟 실패 상태(접근 편차, 헛손질 등)가 발생하면 즉시 안전 정지.
2. 씬을 그대로 유지한 채 `teleop_before_episode=true` 옵션으로 레코더 구동.
3. 사전 텔레옵으로 리더와 팔로워를 실패 상태에 정렬 후 `Enter` 입력.
4. 실패 상태로부터의 교정 동작(재접근, 파지, 리프트, 배치)을 수행.
5. 성공적으로 교정 완료 시 오른쪽 화살표로 저장.

## Validate Every Session

수집 완료 후 데이터셋 병합 또는 학습 전 반드시 검증:

- 실제 에피소드 개수 및 인덱스 연속성.
- 30 FPS 타임스탬프 및 `top`, `wrist` 2개 카메라 비디오 디코딩 무결성.
- 관절 각도(`observation.state`) 및 액션(`action`)에 NaN/Inf 값 유무.
- 그리퍼 개폐 상태 및 파지 시점 피처 확인.
- 데이터셋 검사 완료 후 `dataset-model-registry.md`에 등록.

