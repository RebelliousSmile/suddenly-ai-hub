---
name: golden-principles
description: Non-negotiable agent guidelines for this project
---

# Golden Principles

Rules the agent must not bypass.

## Structure

- One objective per command, agent, rule, or skill file
- Follow SDLC phase numbering strictly (01–10)
- Naming: slugified, lowercase, underscore-separated (commands); kebab-case (rules)
- All files use `.md` extension; skills use `SKILL.md`

## Simplicity & Scope

- Implement only the requested requirements
- Do not add extra files or features beyond scope
- Prefer the simplest working prompt design
- Remove placeholder content (e.g., `[TODO]`, `[Term 1]`, `[...]`) before saving

## Quality

- Every file must have valid YAML frontmatter with `name:` and `description:`
- Mermaid diagrams must have `---\ntitle: ...\n---` header and must not use `\n`
- No broken `@include` references in `.claude/` files
- Memory files use bullet points, not prose

## Tooling

- Prefer CLI tools (`gh`, `git`) over MCPs when equivalent
- Use `pnpm` instead of `npm`
- Always prefix CLI commands with `rtk` per global CLAUDE.md
