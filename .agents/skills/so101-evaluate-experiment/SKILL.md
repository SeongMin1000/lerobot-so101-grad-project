---
name: so101-evaluate-experiment
description: SO-101 모델과 제어 파라미터를 A-H 구역, 랜덤 5블록 배치, 공식 제한시간 아래에서 반복 평가하고 성·접·집·놓·오·충 코드와 오차·시간을 비교 기록한다. Use for evaluation sheets, success rate, zone tests, A/B comparisons, parameter tuning, failure codes, or experiment summaries.
---

# SO-101 Evaluate Experiment

## 1. Load Context & Evaluation Guidelines

Read before acting:

1. `project/docs/agent-context/project-contract.md`
2. `project/docs/agent-context/graduation-requirements.md`
3. `project/docs/agent-context/dataset-model-registry.md`
4. `project/config/experiment-profiles/active.env`

표준 기록 템플릿: `project/docs/agent-context/evaluation-log-template.csv`

---

## 2. Official Evaluation Criteria (Task 1 & Task 2)

졸업과제 공식 평가 규정:

| 태스크 | 공식 명칭 및 목표 | 제한 시간 | 성공 판정 기준 |
| :--- | :--- | :---: | :--- |
| **Task 1** | **5색 블록 슬롯 분류 및 안착** | **3분 (180초)** | 작업 영역 내 5개 블록(빨강, 노랑, 나무, 초록, 파랑)을 지정된 목표 슬롯에 정확하게 분류하여 안착 |
| **Task 2** | **블록 다단 스태킹 (Stacking)** | **5분 (300초)** | 블록을 수직으로 쌓아 올린 후 외력 없이 **최소 5초간 안정 유지** |

* **단일 가중치(Single-Checkpoint) 원칙**: Task 1과 Task 2는 원칙적으로 동일한 단일 신경망 모델 가중치를 기반으로 평가합니다.

---

## 3. Scene Layout & A~H Zone Protocol

### 3.1 5-블록 랜덤 배치 규칙
1. 작업대(Board)의 **A~H 구역**에 5개 블록을 서로 겹치지 않게 무작위 배치.
2. 극단 구역(B 구역 원거리 외곽, G 구역 근접 정면)을 의도적으로 포함하여 모델의 도달 한계(Reachable Envelope) 검증.
3. 동일한 배치 조건에서 1회차, 2회차 반복 평가(`1회차`, `2회차`로 기록).
4. 각 롤아웃이 종료된 후에는 반드시 씬(Scene)을 새롭게 리셋.

---

## 4. Standard Result Codes (성·접·집·놓·오·충)

평가 시 각 블록 조작 단계별로 **단 1개의 표준 결과 코드**를 부여합니다:

| 코드 | 명칭 | 세부 판정 기준 |
| :---: | :---: | :--- |
| **`성`** | **성공** | 요청된 블록을 정상적으로 파지하여 목표 위치에 안착/스태킹 완료 |
| **`접`** | **접근 실패** | 블록으로 이동했으나 파지 전 **1~2cm 정렬 오차** 등으로 헛손질함 (Approach/Alignment Miss) |
| **`집`** | **집기 실패** | 블록에 정상 도달했으나 파지력 부족, 미끄러짐, 조(Jaw) 간섭 등으로 들어 올리지 못함 (Grasp Failure) |
| **`놓`** | **놓기 실패** | 블록을 집어 이동했으나 슬롯 앞턱 충돌(Lip Collision), 오배치, 쓰러짐 발생 (Placement/Release Failure) |
| **`오`** | **오선택** | 지시된 색상/순서가 아닌 엉뚱한 블록을 집거나 조작함 (Wrong Color Selection) |
| **`충`** | **충돌/중단** | 바닥/슬롯과의 물리적 충돌, 모터 트래킹 와치독 중단, 작업자 비상정지(E-stop) 발동 |

---

## 5. Metrics & Comparison Dimensions

### 5.1 기록 항목 (Row Metrics)
- **일시/모델 ID/데이터셋 ID/Git 커밋 해시**
- **태스크명 및 조작 대상 블록 색상**
- **배치 구역(A~H) 및 `배치_회차`**
- **제어 파라미터**: `ACTIONS_PER_CHUNK`, `CHUNK_SIZE_THRESHOLD`, `MAX_RELATIVE_TARGET`, `MAX_TRACKING_ERROR`
- **결과 코드(`성/접/집/놓/오/충`) 및 소요 시간(초)**
- **오차 거리(cm) 및 오차 방향(좌/우/전/후)**

### 5.2 종합 분석 지표
1. **태스크 완수율 (Full Success Rate)**: 제한 시간 내 5개 블록을 모두 성공한 에피소드 비율 (%).
2. **블록 단위 성공률 (Block-level Success Rate)**: 전체 시도된 개별 블록 중 `성` 코드 비율 ($\frac{\text{성공 블록 수}}{\text{총 시도 블록 수}} \times 100\%$).
3. **실패 코드 분포도**: `접 / 집 / 놓 / 오 / 충` 비율 분석 $\to$ 모델 과소적합(`접`), 데이터 궤적 결함(`놓`), 파라미터 오류(`충`) 등 원인 분리.
4. **구역별 히트맵 (Zone Performance)**: A~H 구역별 성공률 분석.

---

## 6. Execution & Safety Protocol

1. 모든 평가는 통합 비동기 런처 [`run_async_inference.sh`](file:///home/eslab/lerobot/project/scripts/robot/run_async_inference.sh)를 통해 수행합니다:
   ```bash
   POLICY_TYPE=act MODEL_PATH="${CANDIDATE_MODEL}" bash project/scripts/robot/run_async_inference.sh
   ```
2. 작업자 안전 수칙:
   - 시작 전 `START` 프롬프트 확인.
   - 키보드 `Ctrl+C` 또는 물리적 비상정지 스위치 상시 대기.
   - 와치독 스톨 방지를 위해 `MAX_TRACKING_ERROR=35.0`, `TRACKING_ERROR_GRACE_STEPS=10` 확인.
3. 충돌이나 비정상 정지 발생 시 즉시 중단하고 해당 회차를 `충`으로 기록한 후 원인을 `so101-diagnose-runtime`으로 분석.

