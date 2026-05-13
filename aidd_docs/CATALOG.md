# AIDD Framework Catalog

Auto-generated framework content: agents, commands, rules, skills, and templates.

> This file is automatically updated by `aidd`.

## Table of Contents

- [agents](#agents)
- [aidd_docs](#aidd_docs)
  - [aidd_docs/templates](#aidd_docstemplates)
- [commands](#commands)
  - [commands/00_behavior](#commands00_behavior)
  - [commands/01_onboard](#commands01_onboard)
  - [commands/02_context](#commands02_context)
  - [commands/03_plan](#commands03_plan)
  - [commands/04_code](#commands04_code)
  - [commands/05_review](#commands05_review)
  - [commands/06_tests](#commands06_tests)
  - [commands/07_documentation](#commands07_documentation)
  - [commands/08_deploy](#commands08_deploy)
  - [commands/09_refactor](#commands09_refactor)
  - [commands/10_maintenance](#commands10_maintenance)
- [rules](#rules)
  - [rules/01-standards](#rules01-standards)
  - [rules/04-tooling](#rules04-tooling)
- [skills](#skills)
  - [skills/aidd-auto-implement](#skillsaidd-auto-implement)
  - [skills/challenge](#skillschallenge)

---

### `agents`

| File | Installed | Description | Mode |
|------|---|---|---|
| alexia.md | [claude](../.claude/agents/alexia.md) · [cursor](../.cursor/agents/alexia.md) · [copilot](../.github/agents/alexia.agent.md) · [opencode](../.opencode/agents/alexia.md) · [codex](../.codex/agents/alexia.toml) | `Act like the USER to autonomously end-to-end implementation without human intervention` | - |
| claire.md | [claude](../.claude/agents/claire.md) · [cursor](../.cursor/agents/claire.md) · [copilot](../.github/agents/claire.agent.md) · [opencode](../.opencode/agents/claire.md) · [codex](../.codex/agents/claire.toml) | `Clarity challenger — challenges and questions until the request is ultra-clear` | - |
| iris.md | [claude](../.claude/agents/iris.md) · [cursor](../.cursor/agents/iris.md) · [copilot](../.github/agents/iris.agent.md) · [opencode](../.opencode/agents/iris.md) · [codex](../.codex/agents/iris.toml) | `Frontend specialist with 3 modes - implement from Figma, verify UI conformity, validate user journeys.` | - |
| kent.md | [claude](../.claude/agents/kent.md) · [cursor](../.cursor/agents/kent.md) · [copilot](../.github/agents/kent.agent.md) · [opencode](../.opencode/agents/kent.md) · [codex](../.codex/agents/kent.toml) | `Use this agent when explicitly asked to perform test-driven development.` | - |
| martin.md | [claude](../.claude/agents/martin.md) · [cursor](../.cursor/agents/martin.md) · [copilot](../.github/agents/martin.agent.md) · [opencode](../.opencode/agents/martin.md) · [codex](../.codex/agents/martin.toml) | `Every time you need to run a command to ensure code is correct, still builds are that tests pass, you must call this agent.` | - |

### `aidd_docs`

#### `aidd_docs/templates`

| File | Installed | Description |
|------|---|---|
| AGENTS.md | [claude](../CLAUDE.md) · [cursor](../AGENTS.md) · [copilot](../.github/copilot-instructions.md) · [opencode](../AGENTS.md) · [codex](../AGENTS.md) | `AI agent configuration and guidelines` |

### `commands`

#### `commands/00_behavior`

| File | Installed | Description |
|------|---|---|
| auto_accept.md | [claude](../.claude/commands/aidd/00/auto_accept.md) · [cursor](../.cursor/commands/aidd/00/auto_accept.md) · [copilot](../.github/prompts/00-auto-accept.prompt.md) · [opencode](../.opencode/commands/aidd/00/auto_accept.md) | `Auto-accept proposed changes without asking for confirmation.` |

#### `commands/01_onboard`

| File | Installed | Description | Argument Hint |
|------|---|---|---|
| generate_agent.md | [claude](../.claude/commands/aidd/01/generate_agent.md) · [cursor](../.cursor/commands/aidd/01/generate_agent.md) · [copilot](../.github/prompts/01-generate-agent.prompt.md) · [opencode](../.opencode/commands/aidd/01/generate_agent.md) | `Generates a customized agent based on user-defined parameters.` | - |
| generate_architecture.md | [claude](../.claude/commands/aidd/01/generate_architecture.md) · [cursor](../.cursor/commands/aidd/01/generate_architecture.md) · [copilot](../.github/prompts/01-generate-architecture.prompt.md) · [opencode](../.opencode/commands/aidd/01/generate_architecture.md) | `Generate project architecture with agents, skills, coordination diagram, and optional rules/commands for code projects` | `Project description and domain requirements` |
| generate_command.md | [claude](../.claude/commands/aidd/01/generate_command.md) · [cursor](../.cursor/commands/aidd/01/generate_command.md) · [copilot](../.github/prompts/01-generate-command.prompt.md) · [opencode](../.opencode/commands/aidd/01/generate_command.md) | `Generate optimized, action-oriented prompts using best practices and structured template` | `The command details to generate the prompt for` |
| generate_rules.md | [claude](../.claude/commands/aidd/01/generate_rules.md) · [cursor](../.cursor/commands/aidd/01/generate_rules.md) · [copilot](../.github/prompts/01-generate-rules.prompt.md) · [opencode](../.opencode/commands/aidd/01/generate_rules.md) | `Generate or modify coding rules manually or auto-scan the codebase to propose rules` | `Rule topic to write, or 'auto' to scan codebase and propose rules` |
| generate_skill.md | [claude](../.claude/commands/aidd/01/generate_skill.md) · [cursor](../.cursor/commands/aidd/01/generate_skill.md) · [copilot](../.github/prompts/01-generate-skill.prompt.md) · [opencode](../.opencode/commands/aidd/01/generate_skill.md) | `Generate a customized skill based on repeated patterns and user workflows.` | `Description of the workflow to package as a skill` |
| init.md | [claude](../.claude/commands/aidd/01/init.md) · [cursor](../.cursor/commands/aidd/01/init.md) · [copilot](../.github/prompts/01-init.prompt.md) · [opencode](../.opencode/commands/aidd/01/init.md) | `Create or update the memory bank files to reflect the current state of the codebase` | - |
| onboard.md | [claude](../.claude/commands/aidd/01/onboard.md) · [cursor](../.cursor/commands/aidd/01/onboard.md) · [copilot](../.github/prompts/01-onboard.prompt.md) · [opencode](../.opencode/commands/aidd/01/onboard.md) | `Detect project state and tell the user exactly what to run next` | - |

#### `commands/02_context`

| File | Installed | Description | Argument Hint |
|------|---|---|---|
| brainstorm.md | [claude](../.claude/commands/aidd/02/brainstorm.md) · [cursor](../.cursor/commands/aidd/02/brainstorm.md) · [copilot](../.github/prompts/02-brainstorm.prompt.md) · [opencode](../.opencode/commands/aidd/02/brainstorm.md) | `Interactive brainstorming session to clarify and refine feature requests` | - |
| challenge.md | [claude](../.claude/commands/aidd/02/challenge.md) · [cursor](../.cursor/commands/aidd/02/challenge.md) · [copilot](../.github/prompts/02-challenge.prompt.md) · [opencode](../.opencode/commands/aidd/02/challenge.md) | `Rethink and challenge previous work for improvements` | - |
| create_user_stories.md | [claude](../.claude/commands/aidd/02/create_user_stories.md) · [cursor](../.cursor/commands/aidd/02/create_user_stories.md) · [copilot](../.github/prompts/02-create-user-stories.prompt.md) · [opencode](../.opencode/commands/aidd/02/create_user_stories.md) | `Create user stories through iterative questioning` | `[Feature description or requirements for user story generation]` |
| ticket_info.md | [claude](../.claude/commands/aidd/02/ticket_info.md) · [cursor](../.cursor/commands/aidd/02/ticket_info.md) · [copilot](../.github/prompts/02-ticket-info.prompt.md) · [opencode](../.opencode/commands/aidd/02/ticket_info.md) | `Get ticket information from the project's ticketing tool` | `[Ticket URL or number]` |

#### `commands/03_plan`

| File | Installed | Description | Argument Hint |
|------|---|---|---|
| components_behavior.md | [claude](../.claude/commands/aidd/03/components_behavior.md) · [cursor](../.cursor/commands/aidd/03/components_behavior.md) · [copilot](../.github/prompts/03-components-behavior.prompt.md) · [opencode](../.opencode/commands/aidd/03/components_behavior.md) | `Define the expected behavior of frontend components into a state machine format.` | `names of the components to define behavior for.` |
| image_extract_details.md | [claude](../.claude/commands/aidd/03/image_extract_details.md) · [cursor](../.cursor/commands/aidd/03/image_extract_details.md) · [copilot](../.github/prompts/03-image-extract-details.prompt.md) · [opencode](../.opencode/commands/aidd/03/image_extract_details.md) | `Analyze image to identify and extract main components with hierarchical organization` | `the image to analyze` |
| plan.md | [claude](../.claude/commands/aidd/03/plan.md) · [cursor](../.cursor/commands/aidd/03/plan.md) · [copilot](../.github/prompts/03-plan.prompt.md) · [opencode](../.opencode/commands/aidd/03/plan.md) | `Generate technical implementation plans from requirements` | `requirements (ticket URL or raw text)` |

#### `commands/04_code`

| File | Installed | Description | Argument Hint |
|------|---|---|---|
| assert.md | [claude](../.claude/commands/aidd/04/assert.md) · [cursor](../.cursor/commands/aidd/04/assert.md) · [copilot](../.github/prompts/04-assert.prompt.md) · [opencode](../.opencode/commands/aidd/04/assert.md) | `Assert that a feature must work as intended.` | - |
| assert_architecture.md | [claude](../.claude/commands/aidd/04/assert_architecture.md) · [cursor](../.cursor/commands/aidd/04/assert_architecture.md) · [copilot](../.github/prompts/04-assert-architecture.prompt.md) · [opencode](../.opencode/commands/aidd/04/assert_architecture.md) | `Verify code conforms to architecture diagrams, ADRs, and project structure.` | `[Optional scope to verify (module, service, or layer name)]` |
| assert_frontend.md | [claude](../.claude/commands/aidd/04/assert_frontend.md) · [cursor](../.cursor/commands/aidd/04/assert_frontend.md) · [copilot](../.github/prompts/04-assert-frontend.prompt.md) · [opencode](../.opencode/commands/aidd/04/assert_frontend.md) | `Assert a frontend feature works as intended.` | `The frontend behavior you need to assert and validate.` |
| implement.md | [claude](../.claude/commands/aidd/04/implement.md) · [cursor](../.cursor/commands/aidd/04/implement.md) · [copilot](../.github/prompts/04-implement.prompt.md) · [opencode](../.opencode/commands/aidd/04/implement.md) | `Implement plan following project rules with validation` | `The technical plan to implement` |
| implement_from_design.md | [claude](../.claude/commands/aidd/04/implement_from_design.md) · [cursor](../.cursor/commands/aidd/04/implement_from_design.md) · [copilot](../.github/prompts/04-implement-from-design.prompt.md) · [opencode](../.opencode/commands/aidd/04/implement_from_design.md) | `Implement a frontend component from a Figma design with pixel-perfect accuracy.` | `The Figma file URL and frame/component to implement.` |
| run_projection.md | [claude](../.claude/commands/aidd/04/run_projection.md) · [cursor](../.cursor/commands/aidd/04/run_projection.md) · [copilot](../.github/prompts/04-run-projection.prompt.md) · [opencode](../.opencode/commands/aidd/04/run_projection.md) | `Project the solution you mentioned on a part of the codebase so we can see if this will work.` | - |

#### `commands/05_review`

| File | Installed | Description | Argument Hint |
|------|---|---|---|
| review_code.md | [claude](../.claude/commands/aidd/05/review_code.md) · [cursor](../.cursor/commands/aidd/05/review_code.md) · [copilot](../.github/prompts/05-review-code.prompt.md) · [opencode](../.opencode/commands/aidd/05/review_code.md) | `Ensure code quality and rules compliance` | - |
| review_functional.md | [claude](../.claude/commands/aidd/05/review_functional.md) · [cursor](../.cursor/commands/aidd/05/review_functional.md) · [copilot](../.github/prompts/05-review-functional.prompt.md) · [opencode](../.opencode/commands/aidd/05/review_functional.md) | `Review feature behavior against plan specification and current diff` | `Plan path to validate against` |

#### `commands/06_tests`

| File | Installed | Description | Argument Hint |
|------|---|---|---|
| test.md | [claude](../.claude/commands/aidd/06/test.md) · [cursor](../.cursor/commands/aidd/06/test.md) · [copilot](../.github/prompts/06-test.prompt.md) · [opencode](../.opencode/commands/aidd/06/test.md) | `List untested behaviors and iterate on test creation until tests pass with best practices` | `[things you want to test]` |
| test_journey.md | [claude](../.claude/commands/aidd/06/test_journey.md) · [cursor](../.cursor/commands/aidd/06/test_journey.md) · [copilot](../.github/prompts/06-test-journey.prompt.md) · [opencode](../.opencode/commands/aidd/06/test_journey.md) | `Test a user journey end-to-end by navigating and validating each step in the browser.` | `The user journey steps to validate and the URL to test on.` |

#### `commands/07_documentation`

| File | Installed | Description |
|------|---|---|
| learn.md | [claude](../.claude/commands/aidd/07/learn.md) · [cursor](../.cursor/commands/aidd/07/learn.md) · [copilot](../.github/prompts/07-learn.prompt.md) · [opencode](../.opencode/commands/aidd/07/learn.md) | `Update memory bank or rules with new information or requirements.` |
| mermaid.md | [claude](../.claude/commands/aidd/07/mermaid.md) · [cursor](../.cursor/commands/aidd/07/mermaid.md) · [copilot](../.github/prompts/07-mermaid.prompt.md) · [opencode](../.opencode/commands/aidd/07/mermaid.md) | `When need to generate Mermaid diagrams` |

#### `commands/08_deploy`

| File | Installed | Description | Argument Hint |
|------|---|---|---|
| commit.md | [claude](../.claude/commands/aidd/08/commit.md) · [cursor](../.cursor/commands/aidd/08/commit.md) · [copilot](../.github/prompts/08-commit.prompt.md) · [opencode](../.opencode/commands/aidd/08/commit.md) | `Create git commit with proper message format` | `auto` |
| create_request.md | [claude](../.claude/commands/aidd/08/create_request.md) · [cursor](../.cursor/commands/aidd/08/create_request.md) · [copilot](../.github/prompts/08-create-request.prompt.md) · [opencode](../.opencode/commands/aidd/08/create_request.md) | `Create PR (GitHub) or MR (GitLab) with filled template` | - |
| tag.md | [claude](../.claude/commands/aidd/08/tag.md) · [cursor](../.cursor/commands/aidd/08/tag.md) · [copilot](../.github/prompts/08-tag.prompt.md) · [opencode](../.opencode/commands/aidd/08/tag.md) | `Create and push git tag with semantic versioning` | - |

#### `commands/09_refactor`

| File | Installed | Description | Argument Hint |
|------|---|---|---|
| audit.md | [claude](../.claude/commands/aidd/09/audit.md) · [cursor](../.cursor/commands/aidd/09/audit.md) · [copilot](../.github/prompts/09-audit.prompt.md) · [opencode](../.opencode/commands/aidd/09/audit.md) | `Perform deep codebase analysis for technical debt and improvements` | `Scope to audit (optional - defaults to full codebase)` |
| performance.md | [claude](../.claude/commands/aidd/09/performance.md) · [cursor](../.cursor/commands/aidd/09/performance.md) · [copilot](../.github/prompts/09-performance.prompt.md) · [opencode](../.opencode/commands/aidd/09/performance.md) | `Optimize code for better performance` | - |
| security_refactor.md | [claude](../.claude/commands/aidd/09/security_refactor.md) · [cursor](../.cursor/commands/aidd/09/security_refactor.md) · [copilot](../.github/prompts/09-security-refactor.prompt.md) · [opencode](../.opencode/commands/aidd/09/security_refactor.md) | `Identify and fix security vulnerabilities` | - |

#### `commands/10_maintenance`

| File | Installed | Description | Argument Hint |
|------|---|---|---|
| debug.md | [claude](../.claude/commands/aidd/10/debug.md) · [cursor](../.cursor/commands/aidd/10/debug.md) · [copilot](../.github/prompts/10-debug.prompt.md) · [opencode](../.opencode/commands/aidd/10/debug.md) | `Debug issue to find root cause.` | - |
| new_issue.md | [claude](../.claude/commands/aidd/10/new_issue.md) · [cursor](../.cursor/commands/aidd/10/new_issue.md) · [copilot](../.github/prompts/10-new-issue.prompt.md) · [opencode](../.opencode/commands/aidd/10/new_issue.md) | `Create issues in the configured ticketing tool` | `Describe the problem you want to create an issue for` |
| reflect_issue.md | [claude](../.claude/commands/aidd/10/reflect_issue.md) · [cursor](../.cursor/commands/aidd/10/reflect_issue.md) · [copilot](../.github/prompts/10-reflect-issue.prompt.md) · [opencode](../.opencode/commands/aidd/10/reflect_issue.md) | `Reflect on possible sources, identify most likely causes, add validation logs before fixing` | - |
| reproduce.md | [claude](../.claude/commands/aidd/10/reproduce.md) · [cursor](../.cursor/commands/aidd/10/reproduce.md) · [copilot](../.github/prompts/10-reproduce.prompt.md) · [opencode](../.opencode/commands/aidd/10/reproduce.md) | `Fix bugs with test-driven workflow from issue to PR` | `Bug description or issue number` |

### `rules`

#### `rules/01-standards`

| File | Installed | Description | Paths | AlwaysApply | Globs | ApplyTo |
|------|---|---|---|---|---|---|
| 1-command-structure.md | [claude](../.claude/rules/01-standards/1-command-structure.md) · [cursor](../.cursor/rules/01-standards/1-command-structure.mdc) · [copilot](../.github/instructions/01-command-structure.instructions.md) · [opencode](../.opencode/rules/01-standards/1-command-structure.md) | `Standards for naming, organizing, and writing command files. Apply when creating or editing any command file.` | - | - | - | - |
| 1-mermaid.md | [claude](../.claude/rules/01-standards/1-mermaid.md) · [cursor](../.cursor/rules/01-standards/1-mermaid.mdc) · [copilot](../.github/instructions/01-mermaid.instructions.md) · [opencode](../.opencode/rules/01-standards/1-mermaid.md) | - | `**/*.mmd,**/*.md` | - | - | - |
| 1-rule-structure.md | [claude](../.claude/rules/01-standards/1-rule-structure.md) · [cursor](../.cursor/rules/01-standards/1-rule-structure.mdc) · [copilot](../.github/instructions/01-rule-structure.instructions.md) · [opencode](../.opencode/rules/01-standards/1-rule-structure.md) | - | `.claude/rules/**/*.md,.claude/rules/**/*.mdc` | - | - | - |
| 1-rule-writing.md | [claude](../.claude/rules/01-standards/1-rule-writing.md) · [cursor](../.cursor/rules/01-standards/1-rule-writing.mdc) · [copilot](../.github/instructions/01-rule-writing.instructions.md) · [opencode](../.opencode/rules/01-standards/1-rule-writing.md) | - | `.claude/rules/**/*.md,.claude/rules/**/*.mdc` | - | - | - |

#### `rules/04-tooling`

| File | Description | AlwaysApply |
|------|---|---|
| [ide-mapping.claude.md](../.claude/rules/04-tooling/ide-mapping.md) | `Claude Code file locations, syntax, frontmatter, and include patterns reference. Apply when creating or configuring Claude-specific files.` | - |
| [ide-mapping.cursor.md](../.cursor/rules/04-tooling/ide-mapping.mdc) | `Cursor file locations, syntax, frontmatter, and rule patterns reference. Apply when creating or configuring Cursor-specific files.` | `false` |
| [ide-mapping.opencode.md](../.opencode/rules/04-tooling/ide-mapping.md) | `OpenCode file locations, syntax, frontmatter, and configuration reference. Apply when creating or configuring OpenCode-specific files.` | - |

### `skills`

#### `skills/aidd-auto-implement`

| File | Installed | Description | Argument Hint |
|------|---|---|---|
| SKILL.md | [claude](../.claude/skills/aidd-auto-implement/SKILL.md) · [cursor](../.cursor/skills/aidd-auto-implement/SKILL.md) · [copilot](../.github/skills/aidd-auto-implement/SKILL.md) · [opencode](../.opencode/skills/aidd-auto-implement/SKILL.md) · [codex](../.agents/skills/aidd-aidd-auto-implement/SKILL.md) | `Autonomously run the AI-Driven Development workflow to code an high quality feature.` | `The URL or file path of the issue or task to implement.` |

#### `skills/challenge`

| File | Installed | Description |
|------|---|---|
| SKILL.md | [claude](../.claude/skills/challenge/SKILL.md) · [cursor](../.cursor/skills/challenge/SKILL.md) · [copilot](../.github/skills/challenge/SKILL.md) · [opencode](../.opencode/skills/challenge/SKILL.md) · [codex](../.agents/skills/aidd-challenge/SKILL.md) | `Review and challenge previous work for improvements and correctness. Use when the user says 'challenge this', 'review my work', 'is this correct', asks for a critical review, or wants to rethink a decision.` |

