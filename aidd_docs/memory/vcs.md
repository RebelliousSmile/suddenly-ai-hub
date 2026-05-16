---
name: vcs
description: Version control system guidelines for suddenly-muses
---

# Versioning Control System (VCS) Guidelines

- Main Branch: `main`
- Platform: GitHub
- Owner: `RebelliousSmile`
- Repo: `RebelliousSmile/suddenly-muses`
- CLI: `gh`
- MCP: `mcp__github__*`
- Ticketing Tool: N/A (GitHub Issues used for roadmap phases)

## Branch Naming Convention

- Format: `type/ticket-short-description`
- Types: `feat/`, `fix/`, `docs/`, `refactor/`, `chore/`, `test/`, `hotfix/`
- No ticket prefix when N/A (e.g. `feat/add-oauth-login`)
- kebab-case, 3-5 words max
- Action verbs: add, fix, update, remove

## Commit Convention

- Format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`, `style`, `ci`, `revert`
- Scope: optional, component affected (e.g. `auth`, `api`, `ui`)
- Description: imperative mood, lowercase, no period, max 72 chars
- Body: explain why, wrap at 72 chars, blank line separator
- Footer: `BREAKING CHANGE:`, `Fixes #123`, `Closes #456`
- Atomic commits — one logical change per commit
