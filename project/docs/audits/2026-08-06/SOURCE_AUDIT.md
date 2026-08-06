# LeRobot 소스 변경·추가·경로 감사 보고서

- 감사일: 2026-08-06
- 기준 소스: `lerobot84.zip`
- 비교 기준: 압축본 내부 Git `HEAD`
- 보조 근거: `hybrid_cv_act_package.zip`, 기존 `PROJECT_COMPLETE_HISTORY_2026-07-28.md`, 이후 대화 기록
- 목적: 지금까지의 변경을 복원하고, 현재 실행에 쓰이는 파일과 과거·백업 파일을 구분하며, 경로와 설정을 일관되게 정리할 수 있는 기준선을 만든다.

## 1. 결론

정리 가능하다. 현재 압축본에는 원본 Git 메타데이터가 보존되어 있어서 **공식 LeRobot 기준으로 수정된 추적 파일 12개**를 정확히 판별할 수 있다. Git에 아직 추가하지 않은 파일도 전수 분류하면 **활성 커스텀 코드 17개**, **설정·자산 11개**, **소스 백업 11개**, **디버그 캡처 384개**다.

다만 “여태껏 존재했던 모든 중간 편집본”을 100% 복원할 수 있다는 뜻은 아니다. 저장하지 않고 덮어쓴 편집, 삭제한 파일, 실제 로봇 PC에만 남은 최신 변경은 압축본이나 대화 기록이 없으면 확정할 수 없다. 따라서 이 문서는 다음처럼 근거를 구분한다.

| 등급 | 의미 |
|---|---|
| A | 현재 압축본의 Git 상태·파일 내용·해시로 확정 |
| B | 현재 코드와 보존된 백업 파일의 직접 비교로 확정 |
| C | 2026-07-28 프로젝트 이력 문서와 대화 기록으로 확인 |
| D | 압축본 이후 대화에서만 논의·시험됨; 현재 파일 반영 여부는 미확정 |

가장 중요한 결과는 다음 다섯 가지다.

1. 현재 소스는 **LeRobot 0.5.2** 기반의 커스텀 포크다.
2. 공식 코드에 섞인 변경은 입력 영상 처리, 비동기 추론 디버그, 로봇 동작 안전 제한, 캘리브레이션, 테스트다.
3. 프로젝트 전용 기능 13개가 `src/lerobot/async_inference/`에 평평하게 섞여 있어 공식 모듈과 경계가 불명확하다.
4. 활성 런타임 JSON이 둘이고 백업도 넷이라, “어느 포즈가 정답인지”가 소스만으로 확정되지 않는다.
5. 즉시 고쳐야 할 문법 오류 1개와, 실제 동작 전 확인해야 할 안전·LoRA·기록 정합성 문제가 있다.

## 2. 기준 스냅샷 식별 정보

| 항목 | 값 | 근거 |
|---|---|---|
| LeRobot 버전 | `0.5.2` | `pyproject.toml`, `PKG-INFO` |
| Git 브랜치 | `main` | 압축본 `.git` |
| 기준 커밋 | `536b9621b2d8b0193be71508140701a18f79c953` | `HEAD`, `origin/main` |
| 기준 커밋 제목 | `Fix pi0fast model id in docs (#3855)` | Git 로그 |
| 원격 저장소 | `https://github.com/huggingface/lerobot.git` | Git remote |
| `lerobot84.zip` SHA-256 | `2365f9734b4e5fc6035815069f3f104efe10b18d5ae1932efd5b1bfcfd094998` | 직접 계산 |
| `hybrid_cv_act_package.zip` SHA-256 | `3ffa8f0433bc074204a01cfc74fcb3ac2ee00da597cda5f5285c8df519f5f0d3` | 직접 계산 |
| 압축본 항목 수 | 2,141 | ZIP 목록 |
| 압축 해제 크기 | 약 331 MiB | 파일시스템 측정 |

`lerobot84`의 `84`는 LeRobot 버전이 아니다.

## 3. Git이 확정해 주는 수정 파일 12개

전체 diff는 **367줄 추가, 64줄 삭제**다.

| 파일 | 증감 | 실제 변경 | 목적·영향 | 상태 |
|---|---:|---|---|---|
| `.gitignore` | `+8/-32` | 프로젝트 디버그·캘리브레이션·백업·추적 출력 무시 규칙 추가 | 생성물의 Git 오염 방지 | 유지하되 생성물은 저장소 밖으로 이동 권장 |
| `src/lerobot/async_inference/configs.py` | `+15/-0` | observation 디버그 저장 경로·개수 설정 추가, 정책 설정으로 전달 | 클라이언트가 서버로 보내는 실제 카메라 배열 점검 | 유지 가능 |
| `src/lerobot/async_inference/helpers.py` | `+61/-23` | HWC→CHW 변환, 선택적 리사이즈, SmolVLA 원본 비율 보존, float 이미지 이중 `/255` 방지, 값 범위 검증, 파일 로그 기본 비활성화 | 입력 영상 왜곡·이중 정규화 해결 | 핵심 수정 |
| `src/lerobot/async_inference/policy_server.py` | `+15/-0` | SmolVLA만 사전 고정 리사이즈 생략, 첫 입력 영상 shape/dtype/min/max 기록 | 모델 processor가 비율 보존 리사이즈를 담당 | 핵심 수정; LoRA 로딩 별도 확인 필요 |
| `src/lerobot/async_inference/robot_client.py` | `+39/-0` | 서버 전송 직전 카메라 배열 PNG·metadata JSON 저장 | 카메라 키·회전·크기 검증 | 진단용; 운영 기본값은 꺼두는 편이 안전 |
| `src/lerobot/robots/so_follower/config_so_follower.py` | `+6/-0` | `max_tracking_error`, `tracking_error_grace_steps` 추가 | 모터 추종 실패 watchdog 설정 | 핵심 안전 수정 |
| `src/lerobot/robots/so_follower/so_follower.py` | `+78/-7` | follower wrist-roll 캘리브레이션 범위를 `1350..3110`으로 변경, 개별 관절 clamp 대신 전체 벡터 공통 비율 제한, 추종 실패 시 현재 자세 hold 후 예외 | 팔 뒤집힘·명령 급변·stall 방지 | 핵심 안전 수정; 제한 대상 재검토 필요 |
| `src/lerobot/robots/utils.py` | `+55/-0` | `ensure_synchronized_goal_position` 추가 | 모든 관절 목표 변화량에 하나의 공통 축척 적용 | 핵심 안전 수정 |
| `src/lerobot/teleoperators/so_leader/so_leader.py` | `+1/-1` | leader wrist-roll 최소 raw 값을 `0`에서 `110`으로 변경 | 리더 캘리브레이션 보정 | 장비 의존 수정 |
| `src/lerobot/utils/utils.py` | `+4/-1` | 로그 파일을 5 MiB, 2개 백업의 회전 로그로 변경 | 무한 로그 증가 방지 | 유지 가능 |
| `tests/async_inference/test_helpers.py` | `+42/-0` | float 이미지 이중 정규화 방지·geometry 보존 테스트 | 입력 수정 회귀 방지 | 유지 |
| `tests/robots/test_so100_follower.py` | `+43/-0` | 공통 축척과 추종 stall hold/abort 테스트 | 안전 수정 회귀 방지 | 유지 |

### 3.1 안전 제한의 현재 실제 동작

현재 구현은 `goal_pos`에 들어온 **모든 모터**를 공통 축척 대상으로 삼는다. 따라서 `wrist_roll`과 `gripper`도 제외되지 않는다. 이후 대화에서는 “팔 뒤집힘을 만들지 않는 두 축은 제외하자”는 방향이 나왔지만, 이는 현재 압축본에는 반영되지 않은 D등급 항목이다.

현재 실행 스크립트 기본값도 다음과 같다.

| 설정 | 압축본 기본값 | 이후 대화의 시험값 | 판정 |
|---|---:|---:|---|
| `MAX_RELATIVE_TARGET` | `1.0` | `0.75`, `5.0` 등 | 시험값은 소스 기본값에 미반영 |
| `ACTIONS_PER_CHUNK` | `10` | `20` | 미반영 |
| `CHUNK_SIZE_THRESHOLD` | `0.5` | `0.6` | 미반영 |
| `AGGREGATE_FN_NAME` | `latest_only` | 주로 동일 | 반영 |
| `MAX_TRACKING_ERROR` | `5.0` | 동일 계열 | 반영 |
| `TRACKING_ERROR_GRACE_STEPS` | `2` | 동일 계열 | 반영 |

## 4. 새로 추가된 활성 커스텀 코드 17개

이 파일들은 Git 기준으로 모두 untracked다. 즉 현재 동작에 중요해도 커밋·버전 이력 보호를 받지 못한다.

| 파일 | 역할 | 현재 분류 | 비고 |
|---|---|---|---|
| `merge_grasp_v2_datasets.py` | 여러 grasp-v2 데이터셋 병합 | 도구, 현재 고장 | 14행 문법 오류로 실행 불가 |
| `scripts/grad_project.sh` | 캘리브레이션·기록·추론 등의 통합 메뉴 | 운영 도구 | 이전 ACT 계열과 현재 흐름이 함께 남아 있음 |
| `scripts/run_smolvla_red_observe_inference.sh` | 로봇 PC SmolVLA observe→추론 실행 | 현재 주 실행기 | 모델·장비·서버·안전 기본값이 한 파일에 집중됨 |
| `scripts/run_smolvla_red_policy_server.sh` | GPU PC 비동기 정책 서버 실행 | 현재 주 실행기 | 모델 경로는 클라이언트가 전달 |
| `src/lerobot/async_inference/hybrid_cv_act_client.py` | OpenCV+ACT 하이브리드 FSM | 레거시 | grasp 성공 검증 없이 시간 경과 후 배치 완료로 처리 |
| `src/lerobot/async_inference/hybrid_cv_diffusion_client.py` | OpenCV+Diffusion 하이브리드 FSM | 레거시 | ACT 버전과 같은 구조적 한계 |
| `src/lerobot/async_inference/hybrid_detect_once.py` | 이미 열린 top 카메라 프레임에서 블록 1회 검출 | 유틸리티 | 카메라 중복 open 회피 |
| `src/lerobot/async_inference/hybrid_goto_both_pose.py` | follower·leader를 같은 저장 포즈로 이동 | 현재 보조 도구 | observe 복귀에 사용 |
| `src/lerobot/async_inference/hybrid_paths.py` | 프로젝트 root·runtime·detector 경로 계산 | 현재 핵심 | 경로 통합의 출발점이나 전 파일이 사용하지는 않음 |
| `src/lerobot/async_inference/hybrid_record_color_sequence.py` | 지정 색 순서 데이터 기록 | 기록 도구 | 클래스 기본 순서는 현재 목표 순서와 다름 |
| `src/lerobot/async_inference/hybrid_record_grasp_zone.py` | 영역별 grasp 데이터 기록 | 기록 도구 | grasp-v2 데이터 생성 계열 |
| `src/lerobot/async_inference/hybrid_save_pose.py` | observe/drop/pregrasp 포즈 저장 | 설정 도구 | 백업 경로를 받지만 실제 백업 생성·반환은 안 됨 |
| `src/lerobot/async_inference/opencv_block_detector.py` | top-view 블록 검출 v4 | 현재 perception 핵심 | ECC, LAB/HSV, workspace, watershed 포함 |
| `src/lerobot/async_inference/opencv_calibrate.py` | detector 배경·색·영역 보정 | 현재 보조 도구 | `/dev/cam_top` 기본값 포함 |
| `src/lerobot/async_inference/opencv_target_verifier.py` | 고정 슬롯별 배치 완료 판정 | 독립 실험 도구 | 현재 추론/FSM 어디에서도 import되지 않음 |
| `src/lerobot/async_inference/safe_config_io.py` | JSON 원자적 저장 | 현재 보조 모듈 | `fsync`+`os.replace`; 백업 기능은 없음 |
| `src/lerobot/async_inference/smolvla_record_observe_return.py` | SmolVLA 전체 pick/place 기록 후 observe 복귀 | 현재 기록 핵심 | 성공 시만 저장, 실패·timeout·Esc 시 discard |

### 4.1 색 순서

프로젝트의 확정 목표 순서는 **red → yellow → wood → green → blue**다. 래퍼 실행에서는 이 순서를 넘기지만, `hybrid_record_color_sequence.py` 클래스 자체 기본값은 아직 **red → wood → yellow → green → blue**다. 모듈을 직접 실행하면 잘못된 순서가 될 수 있다.

### 4.2 현재 SmolVLA 기록기

2026-07-28 초기 설계보다 발전한 end-to-end 자동 기록기다.

- pick, place, release, 짧은 hold까지 데이터에 포함
- 작업 완료·gripper open 상태에서 오른쪽 화살표를 눌러야 저장
- 왼쪽 화살표는 버퍼와 임시 카메라 파일을 완전히 폐기한 뒤 observe 복귀
- timeout은 저장하지 않고 폐기
- `Esc`도 저장하지 않고 정지하며 자동 이동하지 않음
- observe 복귀 구간은 데이터셋에 포함하지 않음

## 5. 설정·자산 11개

| 파일 | 역할 | 판정 |
|---|---|---|
| `assets/opencv/top_background.png` | detector 기준 배경 | 활성 자산 |
| `configs/opencv_detector.json` | detector 파라미터·workspace·색 기준 | 활성 설정 |
| `grasp_v2_splits.json` | grasp 데이터 분할 정보 | 데이터 도구 설정 |
| `hybrid_runtime.json` | 현재 실행 스크립트가 참조하는 런타임 포즈 | 활성 설정이지만 값 확정 필요 |
| `hybrid_runtime_smolvla.json` | 별도 SmolVLA 포즈 설정 | 현재 코드에서 참조되지 않음 |
| `hybrid_runtime.before_clear_pregrasp.json` | 이전 런타임 백업 | 보관 이력 |
| `hybrid_runtime.before_fix_gripper_open.json` | 이전 런타임 백업 | 보관 이력 |
| `hybrid_runtime.before_observe_open.json` | 이전 런타임 백업 | 후보 포즈 포함 |
| `hybrid_runtime.before_wide_paper_20260714_205413.json` | 이전 런타임 백업 | 후보 포즈 포함 |
| `target_empty_reference.png` | 빈 target 기준 이미지 | verifier 자산 |
| `target_verifier.json` | 슬롯·색·임계값 설정 | verifier 설정; 현재 FSM 미연결 |

모든 런타임 JSON에는 pregrasp 8점과 `drop_center`가 있다. 문제는 observe 포즈와 gripper open 값이 서로 다르다는 점이다.

| 파일 | observe 핵심 값 `(pan, lift, elbow, flex, roll, gripper)` | open |
|---|---|---:|
| `hybrid_runtime.before_clear_pregrasp.json` | `(-12.967, -100.879, 49.275, 83.341, -9.670, 0.991)` | `0.991` |
| `hybrid_runtime.before_fix_gripper_open.json` | `(-2.066, -100.791, 28.967, 99.341, -14.330, 59.445)` | `25.429` |
| `hybrid_runtime.before_observe_open.json` | `(-7.692, -101.055, 20.879, 99.516, -1.495, 0.396)` | `22.259` |
| `hybrid_runtime.before_wide_paper_20260714_205413.json` | 위 행과 동일 | `22.259` |
| `hybrid_runtime.json` | `(-0.967, -87.429, 16.659, 95.473, -7.033, 26.852)` | `26.852` |
| `hybrid_runtime_smolvla.json` | `(-8.530, -100.257, 20.886, 97.380, -1.856, 27.244)` | `27.000` |

현재 `run_smolvla_red_observe_inference.sh`가 읽는 것은 `hybrid_runtime.json`이다. 반면 이후 대화에서 다시 언급한 안전 observe 포즈는 `pan=-7.692`, `lift=-101.055` 계열에 가깝다. 실제 로봇에서 검증하기 전에는 어느 쪽을 정답으로 합치면 안 된다.

## 6. 보존된 소스 백업 11개

| 파일 | 의미 |
|---|---|
| `scripts/run_smolvla_red_observe_inference.sh.bak_before_synchronized_safety_20260804_153829` | 동기화 안전 제한 전 실행기 |
| `scripts/run_smolvla_red_observe_inference.sh.pre_smolvla_input_fix_20260802_185227` | SmolVLA 입력 수정 전 실행기 |
| `src/lerobot/async_inference/helpers.py.pre_aspect_fix` | 비율 보존 수정 전 helper |
| `src/lerobot/async_inference/helpers.py.pre_smolvla_input_fix_20260802_185227` | SmolVLA 입력 수정 전 helper; 위 파일과 내용 동일 |
| `src/lerobot/async_inference/policy_server.py.pre_smolvla_input_fix_20260802_185227` | 입력 수정 전 서버 |
| `src/lerobot/async_inference/opencv_target_verifier.py.bak_before_single_slot_fix` | 단일 슬롯 수정 전 verifier |
| `src/lerobot/async_inference/opencv_target_verifier.py.bak_before_slot_fix` | 슬롯 수정 전 verifier |
| `src/lerobot/robots/so_follower/config_so_follower.py.bak_before_synchronized_safety_20260804_153829` | 안전 설정 전 follower config |
| `src/lerobot/robots/so_follower/so_follower.py.bak_before_synchronized_safety_20260804_153829` | 동기화 안전 제한 전 follower |
| `src/lerobot/robots/utils.py.bak_before_synchronized_safety_20260804_153829` | 안전 helper 추가 전 robots utils |
| `tests/robots/test_so100_follower.py.bak_before_synchronized_safety_20260804_153829` | 안전 테스트 추가 전 테스트 |

이 파일들은 Git commit을 대신하는 수동 스냅샷이다. 정리할 때 삭제하기 전에 현재 소스를 커밋하고, 필요한 이력은 Git tag/commit으로 옮겨야 한다.

## 7. 생성물·저장소 비대화

Git에 아직 추가되지 않은 파일은 총 423개다.

| 분류 | 개수 | 설명 |
|---|---:|---|
| `debug/` | 384 | 96회 캡처 × `top.png`, `wrist.png`, `belly.png`, `metadata.json` |
| 활성 코드 | 17 | 4절 파일 |
| 설정·자산 | 11 | 5절 파일 |
| 소스 백업 | 11 | 6절 파일 |

추가로 `.gitignore`에 의해 숨겨진 파일도 401개다.

- `src/` 계열 363개: 대부분 `__pycache__`
- calibration 이미지 25개
- 로그 10개
- output 이미지 2개
- test 계열 생성물 1개

디버그·로그·output·백업을 저장소 root에 계속 두면 “현재 소스가 무엇인지” 판단하기 어려워진다. 소스와 생성물을 물리적으로 분리해야 한다.

## 8. 초기 `hybrid_cv_act_package.zip`과 현재 소스 관계

2026-07-09 초기 패키지에는 Python 소스 4개가 있었다.

- `hybrid_cv_act_client.py`
- `hybrid_detect_once.py`
- `hybrid_save_pose.py`
- `opencv_block_detector.py`

현재 `lerobot84.zip`의 네 파일은 모두 초기 패키지와 내용이 달라졌다. 초기 패키지에는 cwd 기준 `top_detector_calib.json` 경로와 옛 detector CLI `--calib`를 사용하는 스크립트가 남아 있지만, 현재 detector는 `--config` 체계다.

따라서 초기 패키지의 `install_hybrid_files.sh`를 현재 저장소 위에 다시 실행하면 최신 파일을 과거 버전으로 덮어쓸 위험이 있다. 이 ZIP은 **설치 패키지가 아니라 역사 자료**로만 보관해야 한다.

## 9. 경로·설정 일관성 감사

`/home/eslab/...` 같은 사용자 절대 경로가 현재 활성 커스텀 코드에 대량으로 박혀 있지는 않다. 대신 다음과 같은 **기준 혼용**이 문제다.

| 문제 | 현재 예 | 위험 | 정리 방향 |
|---|---|---|---|
| root 기본값 고정 | 세 shell script의 `$HOME/lerobot` | 설치 경로가 다르면 실패 | 스크립트 자신의 위치에서 repo root 계산, `LEROBOT_ROOT`는 override만 허용 |
| 경로 resolver 미통일 | `hybrid_paths.py`는 repo 기준, `safe_config_io.py`의 raw relative path는 cwd 기준 | 실행 위치에 따라 다른 파일을 읽음 | 모든 프로젝트 경로를 하나의 resolver에서 생성 |
| runtime 환경변수 이름 혼용 | Python은 `LEROBOT_RUNTIME_CONFIG`, SmolVLA 스크립트는 `RUNTIME_CONFIG` | 같은 설정을 두 이름으로 관리 | 하나로 통일하고 과거 이름은 deprecation alias |
| 활성 runtime 중복 | `hybrid_runtime.json`, `hybrid_runtime_smolvla.json` | 잘못된 observe 포즈 사용 | 하나의 canonical runtime만 유지, 장비별 override 분리 |
| root 파일과 하위 폴더 혼재 | detector는 `configs/`, runtime/verifier는 root | 파일 위치 추측이 필요 | 모두 `project/config/`로 이동 |
| 자산 위치 혼재 | `assets/opencv/top_background.png`, root의 `target_empty_reference.png` | config 이동 시 참조 깨짐 | 모두 `project/assets/opencv/`로 이동, config 기준 상대 경로 사용 |
| 장비 기본값 고정 | `/dev/so101_*`, `/dev/cam_*` | 다른 PC·udev 이름에서 실패 | `.env` 또는 단일 project config로 모음 |
| 네트워크 기본값 고정 | `100.85.69.64:8080` | GPU PC 주소 변경 시 실패 | `project.env`에서만 관리 |
| 모델 ID 고정 | `eslab1234/smolvla_red_full_126ep_lora_r64_20k_v1` | 새 모델 시험값과 소스가 어긋남 | 명시적 experiment profile 파일로 이동 |
| HF cache 구조 고정 | `~/.cache/huggingface/lerobot/$HF_USER` | cache 위치 변경 시 병합 도구 실패 | LeRobot/HF API 또는 `HF_LEROBOT_HOME` 사용 |
| 생성물 경로가 repo 내부 | `debug/client_camera_inputs` | 압축·백업 크기 급증 | 기본값을 repo 밖의 `var/` 또는 XDG state/cache로 이동 |

## 10. 즉시 해결해야 할 문제

### P0 — 실행 자체를 막는 문제

1. `merge_grasp_v2_datasets.py` 14행에 닫는 괄호가 하나 더 있어 Python 문법 오류가 난다.

### P1 — 실제 로봇 실행 전 확정해야 하는 문제

1. **observe 포즈 기준 파일이 불명확하다.** 현재 실행기는 `hybrid_runtime.json`을 쓰지만 이후 합의 후보 값은 다른 백업과 더 가깝다.
2. **공통 안전 축척이 wrist-roll과 gripper까지 제한한다.** 제외가 의도라면 코드·테스트를 함께 바꿔야 한다.
3. **LoRA adapter-only 저장소 로딩 경로가 불명확하다.** 비동기 `policy_server.py`는 단순히 `policy_class.from_pretrained(...)`를 호출한다. 반면 저장소의 rollout 경로는 `PeftConfig`와 `PeftModel.from_pretrained`를 명시적으로 사용한다. Hub 저장소가 full/merged weights를 포함하면 동작할 수 있지만 adapter만 있다면 비동기 서버에서 실패할 가능성이 있다. 실제 모델 저장소 내용을 확인해야 확정할 수 있다.
4. **기록 action과 물리적으로 전송된 action이 어긋날 수 있다.** 기록기는 teleop의 요청 action을 저장하지만 follower 안전 제한이 실제 전송값을 바꿀 수 있다. 학습 데이터에는 실제 전송·실행된 값이 들어가는지 검증해야 한다.

### P2 — 기능과 유지보수 문제

1. `opencv_target_verifier.py`는 독립 실행만 되고 현재 추론/FSM에 연결되지 않았다.
2. `target_verifier.json`의 `min_component_slot_overlap`은 현재 Python 코드에서 읽지 않는 미사용 키다.
3. `hybrid_runtime_smolvla.json`은 현재 참조되지 않는 orphan 설정이다.
4. `hybrid_save_pose.py`는 백업이 생긴 것처럼 출력할 수 있는 구조지만 실제 writer는 backup을 만들지 않고 `None`을 반환한다.
5. `opencv_target_verifier.py`에는 중복된 설정 코드와 정리되지 않은 주석이 남아 있다.
6. 레거시 ACT/Diffusion FSM은 grasp 성공 확인 없이 시간이 지나면 placed count를 올린다.

## 11. 정적 검증 결과

| 검증 | 결과 |
|---|---|
| Python AST/compile 검사 | `merge_grasp_v2_datasets.py`만 문법 오류, 나머지 검사 대상 통과 |
| 현재 shell script 3개 `bash -n` | 통과 |
| 주요 JSON 5개 파싱 | 통과 |
| `pytest` | 환경에 설치되지 않아 실행 못 함 |
| `ruff`, `shellcheck` | 환경에 설치되지 않아 실행 못 함 |
| 실제 로봇·카메라·GPU 통합 시험 | 수행하지 않음 |

`compileall`은 검사 과정에서 `__pycache__`만 생성했으며 원본 소스는 변경하지 않았다.

## 12. 권장 최종 구조

공식 LeRobot 수정과 졸업과제 전용 코드를 분리하는 것이 핵심이다.

```text
lerobot/
├── src/lerobot/
│   ├── async_inference/              # upstream + 꼭 필요한 공통 수정만
│   ├── robots/                       # follower 안전 수정
│   └── grad_project/                 # 졸업과제 전용 패키지
│       ├── control/
│       ├── inference/
│       ├── perception/
│       ├── recording/
│       ├── tools/
│       └── paths.py
├── project/
│   ├── config/
│   │   ├── runtime.json              # 단 하나의 canonical 값
│   │   ├── detector.json
│   │   ├── target_verifier.json
│   │   └── project.env.example
│   ├── assets/opencv/
│   ├── scripts/
│   │   ├── robot/
│   │   ├── gpu/
│   │   └── tools/
│   └── docs/
│       ├── SOURCE_CHANGELOG.md
│       └── CURRENT_CONFIG.md
├── tests/grad_project/
└── var/                               # Git 제외
    ├── backups/
    ├── debug/
    ├── logs/
    └── outputs/
```

## 13. 안전한 정리 순서

1. **현 상태 동결**: 이 ZIP 해시와 Git 상태를 `audit/2026-08-06` tag 또는 보존 branch로 남긴다.
2. **P0 수정**: 데이터 병합 스크립트 문법 오류를 고치고 최소 dry-run 검증을 추가한다.
3. **실제 장비 기준값 확정**: observe pose, open gripper, 안전 축척 제외 관절, chunk 기본값을 로봇에서 확인한다.
4. **변경을 작은 commit으로 분리**:
   - `fix(async): preserve SmolVLA image geometry`
   - `feat(robot): synchronized target safety and watchdog`
   - `feat(project): add perception and recording tools`
   - `chore(project): centralize paths and configs`
5. **프로젝트 전용 패키지 이동**: `async_inference/`의 `hybrid_*`, `opencv_*`, recorder를 `lerobot.grad_project`로 옮기고 import를 갱신한다.
6. **경로 단일화**: root 자동 탐지 + 하나의 runtime 환경변수 + config-relative assets 규칙으로 통일한다.
7. **레거시 격리**: ACT/Diffusion client와 7월 9일 패키지는 `archive/legacy_act/` 문서 이력으로만 남긴다.
8. **생성물 이동**: debug/log/output/manual backups를 `var/` 또는 저장소 밖으로 옮긴다.
9. **회귀 검증**: unit tests, shell 검사, JSON schema, 카메라 shape, LoRA load smoke test, 무토크/저속 실제 로봇 시험 순서로 검증한다.

## 14. 시기별 변경 지도

| 시기 | 확인된 변화 | 근거 |
|---|---|---|
| 2026-07-09 | 초기 OpenCV+ACT 패키지, cwd 기반 detector 설정, 별도 실행 스크립트 | 초기 ZIP, A/B |
| 2026-07-14~15 | 8개 pregrasp·drop/observe 설정, 경로 resolver, 안전 JSON 저장, detector 설정 체계, carry 중 drop pose의 gripper 제거 | 파일·백업·기존 이력, B/C |
| 2026-07-26 전후 | follower elbow 교체·재보정 이력, follower wrist-roll 범위 축소, leader 최소값 보정 | 코드·기존 이력, A/C |
| 2026-07-28 전후 | SmolVLA 기록 파이프라인, 성공/실패 discard 규칙, observe 복귀 분리 | 현재 recorder·기존 이력, A/C |
| 2026-07-30~08-02 | SmolVLA 런타임 분기, 640×480 비율 보존, 이중 정규화 수정, 입력 디버그 | 코드·백업, A/B |
| 2026-08-04 | 동기화 목표 제한, 추종 watchdog, 카메라 전송본 캡처 | 코드·백업·debug timestamp, A/B |
| 2026-08-05~06 | chunk 20, threshold 0.6, relative target 여러 값, wrist/gripper 제외 방향 등 추가 시험·논의 | 대화만, D |

## 15. 이번 감사가 하지 않은 것

- 원본 소스나 설정을 수정하지 않았다.
- 실제 로봇 PC의 현재 checkout이 이 ZIP과 동일하다고 가정하지 않았다.
- 대화에서 나온 시험값을 자동으로 “최종값”으로 채택하지 않았다.
- 디버그 이미지나 수동 백업을 삭제하지 않았다.
- 모델 Hub 저장소를 열어 LoRA 파일 구성을 확인하지 않았다.

## 16. 다음 작업의 완료 기준

정리가 끝났다고 볼 수 있는 기준은 다음과 같다.

- `git status`에서 필요한 소스·설정이 모두 추적되고 생성물만 무시됨
- 활성 runtime 파일이 하나이며 현재 로봇 검증값과 일치함
- 프로젝트 전용 모듈이 `lerobot.grad_project` 아래에 모임
- 모든 실행기가 어느 작업 디렉터리에서 실행해도 같은 config를 읽음
- 모델 ID·장비 경로·서버 주소·실험 파라미터가 한 profile에서 관리됨
- adapter-only/full/merged 모델 각각의 로딩 규칙이 문서화됨
- 기록 action과 실제 전송 action의 정합성이 테스트로 고정됨
- 초기 패키지와 수동 `.bak` 파일은 Git 이력으로 대체됨

---

이 문서는 2026-08-06에 전달된 압축본을 기준으로 한 **복구 가능한 가장 정확한 소스 기준선**이다. 실제 정리 작업은 이 기준선을 보존한 새 branch에서 수행해야 한다.
