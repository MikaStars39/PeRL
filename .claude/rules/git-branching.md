---
description: Git branching strategy for all repos in this project. Always enforce these rules and warn the user if they are about to violate them.
---

# Git Branching Rules

## PeRL (this repo) & Megatron-LM

- **`main` is protected**: never commit directly to `main`. It only receives merges from feature branches.
- All new work goes on **feature branches** created from `main` (e.g., `feature/xxx`, `fix/xxx`).
- Merge feature branches back to `main` via PR or merge commit.

## slime (fork of THUDM/slime)

- **`main` syncs with upstream only**: never push our own code to `main`.
- **`dev`** is the development branch. All feature branches are created from `dev`.
- `dev` rebases onto `main` regularly to pick up upstream changes.
- Feature branches merge into `dev` (not `main`).
- If contributing upstream, create a branch from `main`, not `dev`.

## Enforcement

- If the user attempts to commit directly on `main`, or create a feature branch from `main` in slime, **warn them before proceeding**.
- When creating commits or branches, always verify which repo and branch you are on first.
