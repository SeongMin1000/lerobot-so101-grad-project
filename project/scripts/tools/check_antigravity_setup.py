#!/usr/bin/env python3
"""Validate the repository-local Antigravity skills and shared context files.

This check is deliberately read-only and uses only the Python standard library.
It does not connect to cameras, motors, the network, Hugging Face, or W&B.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

SKILLS = (
    "so101-record-data",
    "so101-train-policy",
    "so101-run-inference",
    "so101-diagnose-runtime",
    "so101-evaluate-experiment",
    "so101-maintain-project",
    "so101-package-demo",
)

CONTEXT_FILES = (
    "project/docs/agent-context/project-contract.md",
    "project/docs/agent-context/hardware-and-paths.md",
    "project/docs/agent-context/dataset-model-registry.md",
    "project/docs/agent-context/experiment-history.md",
    "project/docs/agent-context/decisions-and-known-failures.md",
    "project/docs/agent-context/graduation-requirements.md",
    "project/docs/agent-context/roadmap.md",
    "project/docs/agent-context/evaluation-log-template.csv",
)

PROFILE_FILES = (
    "project/config/experiment-profiles/active.env.example",
    "project/config/experiment-profiles/active.env",
)

REQUIRED_PROFILE_KEYS = {
    "LEROBOT_ROOT",
    "HF_USER",
    "LEROBOT_RUNTIME_CONFIG",
    "ROBOT_PORT",
    "TELEOP_PORT",
    "TOP_CAM",
    "WRIST_CAM",
    "BELLY_CAM",
    "SERVER_ADDRESS",
    "MODEL_PATH",
    "TASK",
    "CAMERA_KEY_MODE",
    "ACTIONS_PER_CHUNK",
    "CHUNK_SIZE_THRESHOLD",
    "AGGREGATE_FN_NAME",
    "MAX_RELATIVE_TARGET",
    "MAX_TRACKING_ERROR",
    "TRACKING_ERROR_GRACE_STEPS",
    "SKIP_CONFIRM",
}

FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)
FIELD_RE = re.compile(r"^(?P<key>[a-zA-Z0-9_-]+):\s*(?P<value>.+?)\s*$")
EXPORT_RE = re.compile(r"^\s*export\s+([A-Z][A-Z0-9_]*)=", re.MULTILINE)
REFERENCE_RE = re.compile(
    r"`((?:project/docs/agent-context|project/config/experiment-profiles)/[^`]+)`"
)
SECRET_NAME_RE = re.compile(r"(?:TOKEN|PASSWORD|SECRET|API_KEY|PRIVATE_KEY)")
VALID_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ValidationError(RuntimeError):
    pass


def require_file(relative_path: str) -> Path:
    path = ROOT / relative_path
    if not path.is_file():
        raise ValidationError(f"required file is missing: {relative_path}")
    return path


def parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if match is None:
        raise ValidationError(f"invalid or missing YAML frontmatter: {path.relative_to(ROOT)}")

    fields: dict[str, str] = {}
    for line in match.group("body").splitlines():
        field = FIELD_RE.match(line)
        if field is None:
            raise ValidationError(f"unsupported frontmatter line in {path.relative_to(ROOT)}: {line}")
        fields[field.group("key")] = field.group("value")
    return fields


def validate_skill(name: str) -> int:
    relative = Path(".agents/skills") / name / "SKILL.md"
    path = require_file(str(relative))
    fields = parse_frontmatter(path)

    if set(fields) != {"name", "description"}:
        raise ValidationError(
            f"frontmatter must contain only name and description: {relative}"
        )
    if fields["name"] != name:
        raise ValidationError(f"skill name does not match folder: {relative}")
    if not VALID_NAME_RE.fullmatch(name) or len(name) > 64:
        raise ValidationError(f"invalid skill name: {name}")
    if len(fields["description"]) < 40:
        raise ValidationError(f"skill description is too vague: {relative}")

    text = path.read_text(encoding="utf-8")
    if len(text.splitlines()) > 500:
        raise ValidationError(f"SKILL.md exceeds 500 lines: {relative}")

    for reference in REFERENCE_RE.findall(text):
        if "*" in reference or "$" in reference:
            continue
        require_file(reference)

    unexpected = [item.name for item in path.parent.iterdir() if item.name.lower() == "readme.md"]
    if unexpected:
        raise ValidationError(f"extraneous README in skill folder: {relative.parent}")

    print(f"[PASS] skill {name}")
    return 1


def validate_profile(relative_path: str) -> int:
    path = require_file(relative_path)
    result = subprocess.run(
        ["bash", "-n", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValidationError(f"shell syntax failed for {relative_path}: {detail}")

    text = path.read_text(encoding="utf-8")
    keys = set(EXPORT_RE.findall(text))
    missing = sorted(REQUIRED_PROFILE_KEYS - keys)
    if missing:
        raise ValidationError(f"profile is missing keys {missing}: {relative_path}")

    secret_keys = sorted(key for key in keys if SECRET_NAME_RE.search(key))
    if secret_keys:
        raise ValidationError(f"secret-like assignments are forbidden: {secret_keys}")

    print(f"[PASS] profile {relative_path}")
    return 1


def main() -> int:
    checks = 0
    require_file("AGENTS.md")
    checks += 1
    print("[PASS] root AGENTS.md")

    for context_file in CONTEXT_FILES:
        require_file(context_file)
        checks += 1
        print(f"[PASS] context {context_file}")

    for skill in SKILLS:
        checks += validate_skill(skill)

    for profile in PROFILE_FILES:
        checks += validate_profile(profile)

    gitignore = require_file(".gitignore").read_text(encoding="utf-8")
    expected_ignore = "/project/config/experiment-profiles/active.env"
    if expected_ignore not in gitignore:
        raise ValidationError(f"missing .gitignore rule: {expected_ignore}")
    checks += 1
    print("[PASS] active.env ignore rule")

    print(f"[DONE] Antigravity setup validation passed ({checks} checks)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
