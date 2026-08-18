# Official Graduation-Project Requirements

Reviewed from the supplied PDF files and additional clarification text on
2026-08-06.

## Document precedence

The April v1.5 notice and July v2.0 details conflict. Use the newer v2.0 details
for current planning, then re-check the newest course announcement immediately
before the final demo. Do not hide a conflict.

| Item | v1.5 notice | v2.0 details | Current planning rule |
| --- | --- | --- | --- |
| Block size | `5 x 5 x 2 cm` | supplied `4 x 4 x 2 cm` | Use supplied `4 x 4 x 2 cm` blocks |
| Task 1 time | combined/older 5-minute wording | 3 minutes | 3 minutes |
| Task 2 time | 5 minutes | 5 minutes | 5 minutes |
| Task separation | Task 1 then Task 2 | evaluator resets and scores each separately | Treat as separate trials |

## Task 1: place in target area

- Move five differently colored blocks into a target area whose **inside**
  dimensions are `20 cm x 10 cm`.
- Time limit: 3 minutes.
- Score uses the number of blocks in the area; completing all five earns a time
  bonus. Exact point weights were not yet announced in the supplied material.
- A block may touch/cross the boundary by at most 2 cm.
- A block may stand vertically and blocks may partially overlap.
- A stack does not count as valid Task 1 placement.

The team's fixed color slots and color-priority order are implementation
choices, not official Task 1 requirements.

## Task 2: stack

- Stack five blocks vertically in the designated area.
- Time limit: 5 minutes.
- Higher completed stack height receives greater weight.
- Completing all five earns a time bonus.
- The stack must remain stable for at least 5 seconds.

## Initial scene and arena

- All five blocks begin flat on their broad `4 x 4 cm` faces.
- Blocks begin outside the target area.
- Blocks do not overlap initially, but may touch side by side.
- The evaluator chooses random reachable positions; all teams use the same
  initial placement.
- A block can be placed beyond the illustrated A3 boundary if the arm can
  physically reach it.
- The provided checkerboard will be available. A custom board may be placed on
  top of it; board size is not restricted in the clarification.
- Target boundary thickness may be up to 2 cm.
- The target is positioned so its near reference is about 25 cm from the robot
  end, excluding the camera, as illustrated in v2.0.
- Camera count is unrestricted.

## Model and control rules

- One neural-network model/checkpoint must support both Task 1 and Task 2.
- Using a separate neural model per task can be penalized.
- Inputs and outputs of that one neural model are unrestricted.
- Rule-based solutions may differ by task.
- Neural and rule-based components may be combined.
- Imitation learning, reinforcement learning, and other control methods are
  allowed.

The additional supplied clarification text is the strongest explicit source
for the one-model rule.

## Evaluation flow

1. Evaluator places all blocks for Task 1.
2. Team runs Task 1.
3. Evaluator scores Task 1.
4. Evaluator resets all five blocks.
5. Team runs Task 2.
6. Evaluator scores Task 2.

Task 2 is not limited to blocks successfully placed during Task 1.

## Jetson, server, and Docker

- Use the assigned Jetson Orin Nano on the robot side.
- Communicate with the provided evaluation server.
- ROS, TCP, UDP, or a combination is allowed.
- Develop/train on the team's server, then create a Docker image that can be
  transferred to the evaluation server and started without lengthy setup.
- The planned evaluation-server driver is `nvidia-driver-580-open`.

Do not assume the driver statement alone determines the CUDA base image. Verify
the actual evaluation server, NVIDIA container runtime, network policy, volume
mounts, and GPU compatibility before freezing the image.

## Open items to re-confirm

- Exact scoring weights for block count, stack height, and time bonus.
- Final evaluation-server CUDA/container-runtime details.
- Any announcement newer than v2.0.
- Whether network access to Hugging Face is available during evaluation; plan
  for offline model availability unless confirmed otherwise.
