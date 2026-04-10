# Contributing

## Setup

```bash
./scripts/dev-setup.sh
```

This installs dependencies, git hooks (pre-commit, commit-msg, post-commit, pre-push), and verifies your environment. Run it once after cloning. Safe to re-run.

## Workflow

1. Branch off main: `git checkout -b feat/description` (or `fix/`, `chore/`, `docs/`)
2. Commit freely on the branch — no format enforced.
3. Tests: `uv run pytest`
4. Merge to main: `git checkout main && git merge --squash feat/... && git commit`
   - The squash commit **must** be a conventional commit (`feat:`, `fix:`, `chore:`, etc.) — the hook enforces this.
   - A post-commit hook runs `semantic-release` automatically on main to bump the version, update the changelog, and tag.
5. Push: `git push origin main --tags`
6. Cleanup: `git branch -D feat/...`

## Quick reference

| Task | Command |
|------|---------|
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check src/ tests/` |
| Type check | `uv run mypy src/` |
| Run server | `uv run personal-kb` |
