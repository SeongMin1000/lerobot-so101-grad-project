---
name: so101-package-demo
description: SO-101 졸업과제의 Jetson Orin Nano 클라이언트, GPU 평가 서버 Docker 이미지, 오프라인 모델 배포, Task 1/2 단일-checkpoint 리허설과 최종 데모 패키징을 설계·검증한다. Use for Jetson, Docker, evaluation server, CUDA/driver compatibility, deployment, final demo, or submission rehearsal.
---

# SO-101 Package Demo

## Load official and project constraints

Read:

1. `project/docs/agent-context/graduation-requirements.md`
2. `project/docs/agent-context/project-contract.md`
3. `project/docs/agent-context/hardware-and-paths.md`
4. `project/docs/agent-context/roadmap.md`
5. `project/docs/agent-context/dataset-model-registry.md`

Stop and ask if a newer official announcement conflicts with these references.

## Preserve the required architecture

- One SO-101 arm and assigned Jetson Orin Nano on the robot side.
- Evaluation GPU server runs the model container.
- One neural-network checkpoint handles both Task 1 and Task 2.
- Task-specific task text, OpenCV, FSM, verification, and safety logic may vary.
- Communication may use ROS, TCP, UDP, or a combination; current project uses
  the LeRobot async server/client path.

Do not solve deployment by shipping two task-specific neural checkpoints.

## Freeze a reproducible release input

Before building an image, identify:

- exact source commit and required local modifications;
- Python/LeRobot version and lock/dependency files;
- active checkpoint type: full, merged, or adapter plus base;
- preprocessor/postprocessor and stats files;
- task strings and camera-key mapping;
- server port and health check;
- Jetson/client source and device configuration;
- model license and external artifact obligations.

Do not build from an unexplained dirty tree. Preserve a patch or commit for
intentional local changes.

## Design the GPU-server image

Require:

- base image compatible with the actual evaluation host driver, CUDA, and
  NVIDIA container runtime;
- pinned Python dependencies;
- project source installed at the frozen revision;
- checkpoint available without assuming live Hugging Face access;
- non-root runtime where practical;
- explicit model/cache volume strategy;
- `0.0.0.0:8080` server bind or the approved port;
- deterministic entrypoint and readable logs;
- health/readiness check that does not move hardware;
- graceful signal handling;
- no tokens or SSH keys baked into layers.

The supplied document says `nvidia-driver-580-open` is planned, but this does
not by itself select a CUDA image. Verify the real server before the release
freeze.

## Design the Jetson/client deployment

Verify on Jetson:

- stable `/dev/so101_follower`, `/dev/so101_leader`, and camera aliases;
- 640x480@30 MJPG for all cameras and belly 180-degree rotation;
- CPU/GPU camera decode and preprocessing throughput;
- network route and latency to the evaluation server;
- control-loop FPS, queue behavior, coordinated limiter, and watchdog;
- local active profile without secrets;
- physical `START`, Ctrl+C, emergency stop, and torque behavior;
- recovery after server/client reconnect.

Do not assume the current x86 robot-PC environment reproduces on ARM64.

## Validate the container in stages

1. Build with network access in development.
2. Start and smoke-load the model without robot hardware.
3. Run a synthetic observation through preprocessing and action prediction.
4. Start with network disabled to prove model/dependencies are local.
5. Run on a clean GPU host with the target driver/runtime.
6. Connect the Jetson with robot torque disabled or a supported dry-run mode.
7. Perform a bounded physical smoke rollout with full safety confirmation.
8. Export/import the image exactly as planned for evaluation.
9. Repeat on a fresh container, not the original development container.

Record image digest/size, startup time, model-load time, first-action latency,
steady FPS, VRAM, and failure recovery.

## Final Task 1/2 rehearsal

Use official conditions:

- supplied 4x4x2 cm blocks;
- target inside size 20x10 cm;
- random reachable, non-overlapping initial blocks outside target;
- separate evaluator reset between tasks;
- Task 1 limit 3 minutes;
- Task 2 limit 5 minutes and 5-second stable stack;
- same checkpoint for both tasks.

Test:

- cold image/container startup;
- cold and warm client connections;
- all three cameras;
- wrong-color prevention and priority FSM;
- release and target verification;
- server disconnect/reconnect;
- camera disconnect handling;
- watchdog and emergency-stop recovery;
- offline operation;
- log collection for judging and presentation evidence.

## Handoff package

Produce only what the user requests, but a complete release normally includes:

- Docker image or reproducible Dockerfile/build context;
- exact image digest and import/run commands;
- model/checkpoint manifest;
- Jetson setup and launch commands;
- non-secret profile template;
- port/device/volume table;
- smoke-test and final-rehearsal results;
- rollback image/profile;
- known limitations and emergency procedure.

Never publish an image, upload a model, or mutate the evaluation server without
explicit authorization.
