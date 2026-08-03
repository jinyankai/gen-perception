#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/jinyankai/gen-perception}"
REMOTE_URL="${REMOTE_URL:-https://github.com/jinyankai/gen-perception.git}"
TARGET_BRANCH="${TARGET_BRANCH:-codex/stage1-core}"

cd "$PROJECT_ROOT"

if [[ ! -d .git ]]; then
  git init
fi

if git remote get-url origin >/dev/null 2>&1; then
  actual_remote="$(git remote get-url origin)"
  if [[ "$actual_remote" != "$REMOTE_URL" ]]; then
    echo "Refusing to replace unexpected origin: $actual_remote" >&2
    exit 2
  fi
else
  git remote add origin "$REMOTE_URL"
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Refusing to synchronize because tracked changes are present." >&2
  git status --short
  exit 3
fi

git fetch --prune origin "$TARGET_BRANCH"

if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
  git checkout "$TARGET_BRANCH"
  git merge --ff-only "origin/$TARGET_BRANCH"
else
  git checkout -b "$TARGET_BRANCH" --track "origin/$TARGET_BRANCH"
fi

echo "SYNC_RESULT=ok"
git status --short --branch
git log -2 --oneline
python evals/smoke_eval.py
