---
name: so101-maintain-project
description: LeRobot 졸업과제 소스의 디렉터리 이동, import·JSON·셸 경로 수정, AGENTS/skills 유지, 회귀검사, Git 범위 확인, 백업·생성물 정리를 안전하게 수행한다. Use for refactors, path changes, imports, config moves, scripts, source audits, Git cleanup, or project structure maintenance.
---

# SO-101 Maintain Project

## Load context and inspect local rules

Read:

1. root `AGENTS.md`;
2. `project/docs/agent-context/project-contract.md`;
3. `project/docs/audits/2026-08-06/SOURCE_AUDIT.md`;
4. `project/docs/audits/2026-08-06/CUSTOM_FILE_MANIFEST.csv`.

Inspect `git status --short` before editing. Existing changes belong to the
user. Do not reset, checkout, clean, delete, or stage them blindly.

## Respect module boundaries

- Project-only Python belongs in `src/lerobot/grad_project/`.
- Shared async changes remain in `src/lerobot/async_inference/` only when ACT or
  other policies can reasonably share them.
- Hardware safety changes require tests under `tests/robots/`.
- Config/assets/scripts/docs belong under `project/`.
- Generated content belongs under `var/` and must not be treated as source.
- Legacy clients remain isolated under `inference/legacy/`.

Do not re-run the early `hybrid_cv_act_package.zip` installer.

## Before moving a file

Search all references, including docs, imports, shell commands, JSON asset
paths, tests, packaging, and runtime defaults:

```bash
rg -n "old_name|old/path|old\.module" \
  AGENTS.md project src tests pyproject.toml
```

Check for:

- `from`/`import` module names;
- `python -m` entry points;
- `$LEROBOT_ROOT/...` paths;
- relative paths interpreted from CWD versus repo root;
- environment-variable aliases;
- JSON paths resolved relative to config versus repository;
- references inside test fixtures and launch wrappers.

Use `src/lerobot/grad_project/paths.py` as the canonical project path resolver.
Avoid `/home/eslab/...`, transient `/dev/videoN`, and CWD-dependent defaults.

## Make changes narrowly

- Preserve unrelated user modifications.
- Use atomic/validated config writing for JSON edits.
- Do not silently change runtime values while only reorganizing paths.
- Keep `active.env.example` tracked and `active.env` local/ignored.
- Update `AGENTS.md`, the relevant skill, and agent-context reference only when
  the contract or workflow materially changed.
- Do not duplicate the same long history inside every skill.
- Never include credentials in examples, logs, profiles, or commits.

## Required checks

Run the applicable checks from repository root:

```bash
python project/scripts/tools/check_antigravity_setup.py
python -m compileall -q src/lerobot/grad_project

while IFS= read -r -d '' script; do
  bash -n "$script"
done < <(find project/scripts -type f -name '*.sh' -print0)

python - <<'PY'
import json
from pathlib import Path

for path in Path("project/config").rglob("*.json"):
    json.loads(path.read_text(encoding="utf-8"))
    print("[PASS]", path)
PY
```

When the `lerobot` environment is available:

```bash
bash project/scripts/tools/check_runtime_regressions.sh
pytest -q tests/grad_project/test_runtime_regression_check.py \
  tests/robots/test_so100_follower.py
```

Report hardware/network/motor checks that were skipped.

## Old-reference audit after a move

Search specifically for former project locations:

```bash
rg -n \
  "lerobot\.async_inference\.(hybrid_|opencv_|smolvla_record|safe_config)|src/lerobot/async_inference/(hybrid_|opencv_|smolvla_record|safe_config)|hybrid_runtime\.json|configs/opencv_detector\.json" \
  AGENTS.md project src tests || true
```

Classify matches:

- executable import/path: fix;
- compatibility alias: document and test;
- audit/history text: retain if clearly historical;
- backup/generated content: exclude from active checks.

## Git handoff

For a requested commit/push:

1. Show the exact changed-file scope.
2. Do not use `git add -A` when unrelated files exist.
3. Stage only intended paths.
4. Run checks before commit.
5. Use a descriptive commit message.
6. Push only when explicitly requested and confirm the remote result.

Never delete datasets, model outputs, logs needed for diagnosis, or backups just
to make the tree appear clean. Propose a recoverable archive/retention plan.

## Update the audit trail

When structure or behavior materially changes, update:

- `CUSTOM_FILE_MANIFEST.csv` for file role/state;
- `SOURCE_AUDIT.md` for important risks or resolved items;
- the relevant agent-context reference;
- runtime regression checks if a previously observed failure could recur.

Do not claim the refactor is safe merely because Python compiles. Import,
config, asset, shell, server, camera, and motor paths require separate evidence.
