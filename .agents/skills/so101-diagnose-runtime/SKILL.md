---
name: so101-diagnose-runtime
description: SO-101 ACT 및 SmolVLA 런타임의 파지 오차(1~2cm), 그리퍼 와치독 스톨, 카메라 키 불일치, 조명 민감도, 청크/클램프, 모터 트래킹, 네트워크 중단 원인을 증거 순서대로 진단한다. Use for wrong direction, 1-2cm grasp error, camera failure, KeyError, false stall abort, clamp, watchdog, crash, or unexplained inference behavior.
---

# SO-101 Diagnose Runtime

## 1. Respect Task Scope & Evidence-First Principle

사용자가 "왜 실패했는가?", "원인 진단"을 요청할 때는 **코드를 임의로 수정하거나 로봇을 움직이지 않고, 축적된 로그와 물리적 증거를 바탕으로 원인을 먼저 분석 및 보고**합니다.

Read before acting:
1. `project/docs/agent-context/project-contract.md`
2. `project/docs/agent-context/decisions-and-known-failures.md`
3. `project/docs/agent-context/hardware-and-paths.md`
4. `project/config/experiment-profiles/active.env`

---

## 2. Preserve Evidence First

프로세스를 재부팅하거나 현장을 정리하기 전에 반드시 다음 증거를 확보합니다:
1. **타임스탬프가 포함된 클라이언트 및 GPU 서버 전체 로그**.
2. **사용한 모델 ID, 태스크 프롬프트, 2-카메라 키 매핑, 청크/클램프/와치독 설정값, Git 커밋 해시**.
3. **클라이언트/서버 캡처 이미지 디렉터리** (`var/debug/client_camera_inputs`).
4. **모터 트레이스 CSV** (`var/debug/motor_traces`) 및 마지막 요청/전송/피드백 액션 행.
5. **물리적 실패 증상, 실패 구역(A~H), 오차 방향/거리(cm), 표준 실패 코드(성·접·집·놓·오·충)**.

---

## 3. Pipeline Trace Order (9단계 진단 파이프라인)

오작동 발생 시 아래 9단계를 순서대로 추적하여 **최초로 이상이 발생한 단계(Root Cause)**를 특정합니다:

1. **[Robot PC] 카메라 프레임 캡처**: Top/Wrist 정상 30 FPS, MJPG 640x480 인코딩.
2. **[GPU Server] 텐서 역직렬화**: gRPC 전송 후 수신된 원시 픽셀 텐서.
3. **[Async Helper] 포맷 변환**: Shape `[B, C, H, W]`, dtype `float32`, `[0, 1]` 정규화.
4. **[Preprocessor] 피처 매핑**: `top -> camera1, wrist -> camera2` Rename Processor 및 통계(`stats.json`) 적용.
5. **[Policy Model] 원시 액션 청크 예측**: 모델이 출력한 정규화 텐서 `[B, Chunk, Action_dim]`.
6. **[Postprocessor] 물리 단위 변환**: 정규화 해제 후 실제 로봇 각도(Degrees) 변환.
7. **[Action Queue] 큐 선택 및 청크 소비**: `latest_only` 정책 및 `ACTIONS_PER_CHUNK` 슬라이싱.
8. **[Coordinated Limiter] 속도 클램프**: `MAX_RELATIVE_TARGET` 스텝 리미터 적용 후 실제 모터 전송 명령.
9. **[Motor Feedback] 모터 물리적 추종 및 와치독**: 실제 모터 위치 피드백 및 트래킹 오차 계산.

---

## 4. Symptom Decision Tree & Field Troubleshooting

현장에서 발생하는 7대 주요 고장 증상과 해결책:

### 4.1 🔥 그리퍼 파지 시 모터 트래킹 와치독 중단 (`Motor tracking error exceeded safety limit`)
* **현상**: 로봇이 블록에 도달하여 정상적으로 집는 순간 즉시 롤아웃이 비상 중단됨.
* **원인**: 물리적 블록 두께(약 2cm)로 인해 실제 그리퍼는 `23° ~ 25°`에서 물리적으로 멈춤. 반면 모델은 완전 닫힘(`0° ~ 2°`)을 명령하여 **약 22°의 정상적인 물리 오차**가 발생함. `MAX_TRACKING_ERROR`가 20.0°로 낮으면 이를 모터 스톨(Stall)로 오인하여 강제 중단함.
* **해결책**:
  ```bash
  export MAX_TRACKING_ERROR=35.0
  export TRACKING_ERROR_GRACE_STEPS=10
  ```

---

### 4.2 🎯 1~2cm 미세 파지 실패 (방향/접근은 완벽하나 마지막 블록 파지 미스)
* **현상**: 블록 색상 구분, 대략적인 위치 이동, 슬롯 복귀는 완벽하지만, 마지막에 블록을 집으려 할 때 1~2cm 빗겨나가 헛손질함 (Loss가 0.05 부근에서 더 이상 떨어지지 않음).
* **주요 원인**:
  1. 🚨 **위치 왜곡 이미지 증강(`RandomAffine`) 사용**: LeRobot의 기본 증강에 포함된 `RandomAffine`(회전 ±5°, 이동 ±5%)은 카메라는 움직이지만 액션 라벨은 제자리이므로, 2cm 블록 기준으로 1~2cm의 인위적인 공간 라벨 노이즈를 주입함. 아무리 스텝을 늘려도(25만 스텝) Loss가 0.05에서 정체되고 실물 파지 시 1~2cm 빗겨남.
  2. **Chunk Size 60 확장**: 60스텝(2.0초) 장기 예측은 후반부 누적 분산과 오차가 커서 파지 순간의 위치 정밀도가 떨어짐.
  3. **학습 Epoch 부족(Underfitting)**: 대규모 데이터셋(단일 태스크 45만+ 프레임, 멀티태스크 75만+ 프레임)에서 최소 5~8 Epoch 이상 반복 학습되지 않은 경우.
* **해결책**:
  - **이미지 위치 왜곡 차단**: `--dataset.image_transforms.enable=false` 적용 (330ep 성공 모델의 핵심 세팅. 조명 대응 필요 시 위치 왜곡 없는 `ColorJitter`만 유지).
  - **Chunk Size 50 통일**: `--policy.chunk_size=50 --policy.n_action_steps=50`으로 복원하여 단기 정밀 수렴 유도.
  - **유효 Epochs 확보**: 단일 태스크 15만 스텝(5+ Epochs), 멀티태스크는 태스크당 최소 5 Epochs(총 30만+ 스텝) 확보.

---

### 4.3 📦 타겟 슬롯 안착 시 앞턱 충돌 (Slot Lip Collision)
* **현상**: 블록을 슬롯으로 잘 가져왔으나, 내려놓을 때 컨테이너 앞쪽 턱에 부딪혀 블록이 튕겨 나감.
* **원인**: 데이터 수집 시 수직 상승/하강(Vertical Clearance) 궤적이 누락되었거나 모델이 수평으로 낮게 접근함.
* **해결책**:
  - 데이터 수집 시 **0.6초 수직 상승 + 손목 숙임(Wrist flex down) 궤적**이 포함된 One-Take v3 데이터셋으로 재학습.

---

### 4.4 🔑 `KeyError: observation.images.*` 또는 `observation.state`
* **현상**: 클라이언트 연결 직후 서버에서 KeyError가 발생하며 즉시 종료됨.
* **원인**: 데이터셋 피처 키(`top`, `wrist`)와 모델 체크포인트 입력 키(`camera1`, `camera2`) 불일치.
* **해결책**:
  - `CAMERA_KEY_MODE=policy` 설정 확인.
  - 체크포인트 내부의 `preprocessor.json`에 `top -> camera1, wrist -> camera2` Rename Processor가 정상 저장되어 있는지 점검.

---

### 4.5 ☀️ 조명/그림자 변화에 대한 동작 불안정 (실험실 형광등에선 잘 되는데 햇빛/그림자에 오작동)
* **현상**: 특정 조명이나 특정 시간대에만 블록을 인식하지 못하고 엉뚱한 곳으로 이동함.
* **원인**: 이미지 증강(Data Augmentation) 없이 학습하여 시각적 과적합(Visual Overfitting) 발생.
* **해결책**:
  - 학습 시 `--dataset.image_transforms.enable=true --dataset.image_transforms.max_num_transforms=3` 플래그를 필수로 적용하여 재학습.

---

### 4.6 🔄 재접속 후 이상한 첫 번째 청크 동작 (Stale Action Chunk)
* **현상**: 추론을 중단했다가 다시 켰을 때 첫 동작이 과거 이전 세션의 동작을 반복함.
* **원인**: GPU 서버의 `Ready()` 핸들러에서 이전 큐가 정상 플러시(Flush)되지 않음.
* **해결책**:
  - `Ready()` 핸들러가 관측 큐, 예측 타임스텝, `last_processed_obs`를 100% 초기화하는지 확인.
  - 서버 측 코드를 수정한 경우 반드시 서버 프로세스를 완전히 재시작.

---

### 4.7 🔌 손목 카메라 프레임 드롭 및 USB 대역폭 오류
* **현상**: 손목 카메라(`cam_wrist`) 영상이 멈추거나 OpenCV 읽기 타임아웃 발생.
* **원인**: USB 허브 대역폭 부족 또는 YUYV 포맷 대역폭 과점.
* **해결책**:
  - 모든 카메라 포맷을 **MJPG 640x480 @ 30 FPS**로 고정.
  - Top 카메라와 Wrist 카메라를 물리적으로 서로 다른 USB 컨트롤러/루트에 분리 연결.

---

## 5. Diagnostic Report Format

진단 보고서 작성 표준 템플릿:

1. **재현된 증상 (Symptom)**: 구체적인 물리적 실패 동작 및 에러 메시지.
2. **통과한 검증 항목 (Passed Evidence)**: 정상 작동이 확인된 파이프라인 단계.
3. **최초 고장 단계 (First Divergence Stage)**: 9단계 중 이상이 처음 관측된 단계.
4. **근본 원인 신뢰도 (Root Cause Confidence)**: `확정(Confirmed)`, `유력(Likely)`, `미결(Unresolved)`.
5. **권장 해결 방안 (Recommended Action)**: 파라미터 조정, 데이터셋 추가, 재학습 등 구체적 조치.

