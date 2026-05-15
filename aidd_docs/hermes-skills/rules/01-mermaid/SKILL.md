---
name: rule-01-mermaid
description: Rule: 01 mermaid
category: rules
---

# Rule: 01 Mermaid

## Description

This rule defines rule: 01 mermaid. It applies to the project context.

## Context

## Header

- Always have title in schema using "---" to define it

## Global

- NEVER use "\n"

## States and nodes

- Define groups, parents, children
- Use fork and join states
- Use clear and concise names
- Use "choice" for conditions
- No standalone nodes
- No empty nodes

## Naming

- Declare elements only (no links) at top
- Consistent naming throughout
- Descriptive names (no "A", "B")
- Node IDs: unquoted alphanumeric (`MyNode`, not `"MyNode"`)
- Labels: in brackets with quotes (`MyNode["My Label"]`)
- Replace ":" with "$" in state names

## Links

- Links made at bottom of diagram
- Use direction when possible
- "A -- text --> B" for regular links
- "A -.-> B" for conditional links
- "A ==> B" for self-loops

## Styles

- Do style unless specified by user

## Gantt

- Use tags: `active`, `done`, `crit`, `milestone`
- Tags are combinable

## Application

When working on this project, always follow these guidelines.
