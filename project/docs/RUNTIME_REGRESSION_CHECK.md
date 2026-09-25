# SmolVLA 런타임 회귀 진단기

이 도구는 추론 결과가 다시 이상해졌을 때 과거의 세 문제가 재발했는지 확인한다.

1. 로봇 PC의 정상 카메라 영상이 GPU에서 흰색 또는 보라색으로 변한 문제
2. 어깨 관절이 목표를 따라가지 못하는 동안 다른 관절만 진행해 팔이 넘어간 문제
3. SmolVLA 입력이 async helper에서 256×256으로 축소된 후 모델에서 512×512로 재확대된 문제

진단기는 카메라나 모터 장치를 열지 않으며 로봇을 움직이지 않는다. 현재 코드 경로를 합성 입력으로 실행하고, 정상 추론 실행이 남긴 최근 카메라·모터 블랙박스 자료를 읽기만 한다.

## 사용법

추론이 이상하면 로봇 클라이언트를 `Ctrl+C`로 종료한다. 종료 과정에서 최근 모터 trace가 저장된다. 그다음 저장소에서 실행한다.

```bash
cd ~/lerobot
conda activate lerobot
bash project/scripts/tools/check_runtime_regressions.sh
```

결과는 기본적으로 다음 경로에 생성된다.

```text
var/diagnostics/regression_YYYYMMDD_HHMMSS/
├── report.md
├── report.json
└── camera_comparison.png
```

종료 코드는 `FAIL`이 하나도 없으면 `0`, 하나라도 있으면 `1`이다. `SKIP`은 필요한 실제 캡처가 없어서 코드 자체만 검사했다는 뜻이며 실패로 처리하지 않는다.

## 실행 중 자동으로 남는 자료

수정된 실행 스크립트를 사용하면 큰 지속 녹화 대신 다음의 작은 자료만 남는다.

- 로봇 PC: 서버로 보내기 직전의 첫 `top/wrist/belly` 프레임
- GPU PC: 같은 capture ID의 `raw/helper/policy` 단계 프레임
- 로봇 PC: 종료 직전 최대 300개의 `요청 목표/실제 전송 목표/현재 관절값`

기본 경로는 다음과 같다.

```text
var/debug/client_camera_inputs/capture_.../
var/debug/server_camera_inputs/capture_.../
var/debug/motor_traces/trace_....json
```

GPU server는 새 디버그 캡처 코드를 읽도록 한 번 재시작해야 한다.

## 두 PC의 영상을 한 번에 비교하는 방법

로봇 PC와 GPU PC는 파일시스템이 다르므로 픽셀 단위 비교에는 양쪽 캡처가 한 PC에 있어야 한다. 두 디렉터리의 이름은 동일한 `capture_...` ID다. 로봇 PC의 해당 디렉터리를 GPU PC의 `var/debug/client_camera_inputs/`로 복사한 뒤 기본 명령을 실행하면 된다.

경로를 직접 지정할 수도 있다.

```bash
bash project/scripts/tools/check_runtime_regressions.sh \
  --client-captures /path/to/client_camera_inputs \
  --server-captures /path/to/server_camera_inputs \
  --motor-traces /path/to/motor_traces
```

## 주요 판정 읽는 법

| 검사 ID | 의미 |
|---|---|
| `camera.code_path` | 직렬화, RGB 채널 순서, `[0,1]` 단일 정규화 코드가 정상인지 |
| `camera.frame.*` | 흰 화면, 거의 단색, 전면 보라색 통계가 검출되는지 |
| `camera.transport.*` | 같은 캡처의 로봇 PC와 GPU raw 픽셀이 완전히 같은지 |
| `motor.code_path` | 전체 목표 벡터 공통 축척과 tracking watchdog이 실제 제어 경로에 있는지 |
| `motor.trace` | 실제 실행에서 관절 추종 오차 또는 관절별 비동기 명령이 발생했는지 |
| `smolvla.preprocessing` | 640×480 → 모델 내부 512×512 단일 resize 코드 경로인지 |
| `smolvla.artifact_geometry.*` | 실제 GPU helper 단계에서 256×256 중간 축소가 발생했는지 |
| `server.process_freshness` | 실행 중 서버가 현재 소스보다 먼저 시작되어 재시작이 필요한지 |

`camera.code_path`, `motor.code_path`, `smolvla.preprocessing`만 PASS이고 실제 자료 검사가 SKIP이라면 “현재 파일의 수정은 살아 있다”까지만 확정할 수 있다. 실제 실행에서 문제가 없었다고 확정하려면 양쪽 카메라 캡처와 모터 trace도 함께 확인해야 한다.
