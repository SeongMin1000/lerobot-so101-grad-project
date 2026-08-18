---
name: so101-evaluate-experiment
description: SO-101 모델과 제어 파라미터를 A-H 구역, 랜덤 5블록 배치, 공식 제한시간 아래에서 반복 평가하고 성·접·집·놓·오·충 코드와 오차·시간을 비교 기록한다. Use for evaluation sheets, success rate, zone tests, A/B comparisons, parameter tuning, failure codes, or experiment summaries.
---

# SO-101 Evaluate Experiment

## Load context

Read:

1. `project/docs/agent-context/project-contract.md`
2. `project/docs/agent-context/graduation-requirements.md`
3. `project/docs/agent-context/dataset-model-registry.md`
4. `project/config/experiment-profiles/active.env`

Use `project/docs/agent-context/evaluation-log-template.csv` as the canonical
field template. It intentionally has no trial-ID column.

## Define the comparison before rollout

State:

- question/hypothesis;
- baseline and candidate;
- one independent variable;
- controlled variables;
- number and distribution of scenes;
- primary metric and stop/safety rule;
- whether the server/client state is cold or warm.

Do not compare a new model and new chunk/clamp values simultaneously unless the
goal is explicitly an end-to-end system comparison.

## Scene protocol

For spatial red-policy testing:

1. Cover A-H zones.
2. Use at least two independently randomized layouts per required zone batch,
   recorded as `1회차`, `2회차`, not `1`, `2`.
3. Place the red block at the intended test point and randomize the other four
   blocks without overlap.
4. Include B-zone extremes and G-zone center/front-angle points deliberately.
5. Keep camera, board, lighting, observe pose, model, task, and server state
   constant within a parameter comparison.
6. Reset the scene after every rollout.

For final Task 1/2 rehearsal, use evaluator-style random reachable placements
and official separate resets instead of zone-centered convenience layouts.

## Result codes

Use exactly one primary result code:

| Code | Meaning |
| --- | --- |
| `성` | Completed requested task successfully |
| `접` | Failed to approach/alignment before grasp |
| `집` | Reached block but failed grasp/lift |
| `놓` | Grasped block but failed placement/release/stability |
| `오` | Selected or manipulated wrong color |
| `충` | Collision, watchdog, emergency stop, or other safety abort |

Record secondary details rather than inventing more primary codes.

## Measurements

Capture for every row:

- date/run/model/dataset/commit;
- cold or warm server state;
- zone and `배치_회차`;
- task and scene description;
- chunk, threshold, aggregation, clamp, watchdog;
- result code and elapsed seconds;
- approach error distance and direction;
- grasp-angle quality;
- release success;
- safety-stop status;
- concise notes.

Use centimeters only when measured or consistently estimated from a calibrated
reference. Otherwise mark the distance as unknown and retain direction.

## Official timing and success

- Task 1: 3 minutes; target inside size `20 x 10 cm`.
- Task 2: 5 minutes; require 5 seconds of stable stacking.
- Score Task 1 and Task 2 separately after scene reset.
- Use one neural checkpoint for both.

The team's fixed slots and priority order may be evaluated as internal design
requirements, but label them separately from official success.

## Safety during evaluation

Running an evaluation plan does not itself authorize physical motion. Use
`so101-run-inference` for each authorized rollout. Preserve `START`, Ctrl+C,
watchdog, emergency stop, and clear-workspace checks.

Classify a safety stop as data. Never repeat an identical unsafe rollout merely
to increase sample count.

## Analyze

Report both counts and denominators:

- overall success rate;
- per-zone and near-front success rate;
- distribution of `접/집/놓/오/충`;
- median/mean completion time for successes;
- lateral error direction counts;
- cold-versus-warm difference;
- confidence limits or uncertainty when sample size is small.

Do not claim improvement from one or two favorable trials. For a targeted
recovery experiment, require the target failure category to decrease without a
clear regression in nominal scenes.

## Update status

After review:

- append results to a new experiment CSV copied from the template;
- summarize the comparison in `experiment-history.md` only when material;
- update registry status only with verified evidence;
- update `active.env` only after the user chooses the winning profile.
