# SO-101 Antigravity 스킬 오버레이

이 압축 파일은 `lerobot806.zip` 전체를 다시 포함하지 않고, Antigravity
설정을 위해 추가하거나 수정한 파일만 담고 있다. 기존 프로젝트의 대용량
로그, 이미지, 모델 출력은 포함하지 않는다.

## 적용 전 조건

- 기존 `lerobot806.zip` 구조가 `~/lerobot`에 적용되어 있어야 한다.
- 현재 수정 중인 파일이 있다면 먼저 `git status --short`로 확인한다.
- 이 오버레이는 `AGENTS.md`, `.gitignore`, 프로젝트 감사 문서 2개를
  `lerobot806.zip` 기준의 수정본으로 갱신한다. 해당 파일에 별도 로컬
  수정이 있으면 먼저 백업하거나 Git으로 커밋한다.

## 적용

다운로드한 ZIP이 `~/Downloads/so101_antigravity_overlay.zip`에 있다고
가정한다.

```bash
cd ~/lerobot
git status --short

mkdir -p var/backups/antigravity_before_install
cp -a AGENTS.md .gitignore var/backups/antigravity_before_install/

cd ~
unzip -o ~/Downloads/so101_antigravity_overlay.zip

cd ~/lerobot
conda activate lerobot
python project/scripts/tools/check_antigravity_setup.py
```

마지막 검사에서 `19/19 PASS`가 나오면 Antigravity 스킬 구조가 정상이다.

## Antigravity 실행

```bash
cd ~/lerobot
agy
```

Antigravity가 읽을 핵심 위치는 다음과 같다.

- 항상 적용되는 규칙: `AGENTS.md`
- 작업별 스킬: `.agents/skills/so101-*/SKILL.md`
- 현재 실험값: `project/config/experiment-profiles/active.env`
- 프로젝트 이력과 기준: `project/docs/agent-context/`

`active.env`에는 토큰, SSH 키, W&B API key 같은 비밀정보를 넣지 않는다.
