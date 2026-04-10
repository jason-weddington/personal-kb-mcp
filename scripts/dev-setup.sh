#!/usr/bin/env bash
set -euo pipefail

# Dev environment setup for personal-kb.
# Run once after cloning. Safe to re-run anytime.

echo "==> Installing dependencies (uv sync)..."
uv sync

echo "==> Installing pre-commit hooks (all types)..."
# pre-commit refuses to install when core.hooksPath is set, even if it
# points to the default .git/hooks dir. Unset it temporarily if needed.
if git config --get core.hooksPath &>/dev/null; then
  echo "  (unsetting core.hooksPath — pre-commit manages hooks directly)"
  git config --unset-all core.hooksPath
fi
uv run pre-commit install \
  --hook-type pre-commit \
  --hook-type commit-msg \
  --hook-type post-commit \
  --hook-type pre-push

echo "==> Verifying setup..."
errors=0

if ! command -v uv &>/dev/null; then
  echo "  FAIL: uv not found — install from https://docs.astral.sh/uv/"
  errors=$((errors + 1))
else
  echo "  OK: uv $(uv --version | awk '{print $2}')"
fi

if [ -f .git/hooks/post-commit ]; then
  echo "  OK: post-commit hook installed (semantic-release)"
else
  echo "  FAIL: post-commit hook missing"
  errors=$((errors + 1))
fi

if [ -f .git/hooks/commit-msg ]; then
  echo "  OK: commit-msg hook installed (conventional commits)"
else
  echo "  FAIL: commit-msg hook missing"
  errors=$((errors + 1))
fi

if uv run python -c "import personal_kb" 2>/dev/null; then
  echo "  OK: personal-kb importable"
else
  echo "  FAIL: personal-kb not importable"
  errors=$((errors + 1))
fi

if [ $errors -gt 0 ]; then
  echo "==> Setup completed with $errors error(s). See above."
  exit 1
else
  echo "==> Done. Ready to develop."
fi
