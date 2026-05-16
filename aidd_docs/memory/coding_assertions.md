---
name: coding-assertions
description: Code quality verification checklist
---

# Coding Guidelines

> Those rules must be minimal because they MUST be checked after EVERY CODE GENERATION.

## Requirements to complete a feature

**A feature is really completed if ALL of the above are satisfied.**

- Markdown files are valid and well-formed
- JSON config files pass syntax validation
- No broken `@include` references in `.claude/` files
- Frontmatter fields match naming conventions

## Commands to run

### Before commit

N/A - no build system configured

### Before push

N/A - no build system configured
