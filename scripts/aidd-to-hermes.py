#!/usr/bin/env python3
"""
Bridge AIDD Framework -> Hermes Skills

Scanne le dépôt `suddenly-ai-hub`, détecte les éléments AIDD dans `.github/`
et génère des SKILL.md compatibles Hermes dans `aidd_docs/hermes-skills/`.

Covers:
- Agents (.github/agents/*.agent.md)
- Rules (.github/instructions/*.md)
- Skills (.github/skills/**/*.md)
- Prompts (.github/prompts/**/*.md)
- Templates (aidd_docs/templates/**/*.md)

Usage:
  python3 scripts/aidd-to-hermes.py              # Générer tous les éléments
  python3 scripts/aidd-to-hermes.py --list        # Lister les éléments détectés
  python3 scripts/aidd-to-hermes.py --install     # Installer dans ~/.hermes/skills/
  python3 scripts/aidd-to-hermes.py --types agent rule skill            # Générer uniquement certains types
"""

import argparse
import os
import re
import sys
import shutil
from pathlib import Path


# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AIDD_SRC = PROJECT_ROOT / ".github"
AIDD_DOCS = PROJECT_ROOT / "aidd_docs"
HERMES_SKILLS = AIDD_DOCS / "hermes-skills"
HERMES_INSTALL_DIR = Path.home() / ".hermes" / "skills" / "aidd-overlay"


# --- Parsers ---

def parse_frontmatter(content):
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not fm_match:
        return {}, content
    fm_text = fm_match.group(1)
    body = content[fm_match.end():].strip()
    
    metadata = {}
    for line in fm_text.splitlines():
        m = re.match(r"^\s*(\w+):\s*['\"]?(.*?)['\"]?\s*$", line)
        if m:
            metadata[m.group(1)] = m.group(2)
    return metadata, body


def clean_body(body):
    # Remove main H1 if present, handle standard headers
    lines = body.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


# --- Converters ---

def convert_agent(file_path, metadata, body):
    name = metadata.get("name", file_path.stem).lower()
    desc = metadata.get("description", "AIDD Agent")
    
    # Better parsing: keep lines as-is and add "- " prefix when outputting
    rules = []
    workflow = []
    current_list = None
    
    for line in body.splitlines():
        stripped = line.strip()
        # Identify list context by section headers
        if stripped.startswith("## ") and "rule" in stripped.lower():
            current_list = rules
        elif stripped.startswith("## ") and ("step" in stripped.lower() or "workflow" in stripped.lower()):
            current_list = workflow
        elif stripped.startswith("### "):
            current_list = None
            continue
        
        # Detect list items
        if stripped.startswith("- ") or stripped.startswith("* "):
            if current_list is not None:
                current_list.append(stripped)
            else:
                rules.append(stripped)
        elif re.match(r"^\d+\.", stripped):
            if current_list is not None:
                current_list.append(stripped)
            else:
                workflow.append(stripped)
    
    # Fallbacks
    if not rules:
        rules = [l for l in body.splitlines() if l.strip() and not l.startswith("#")][:10]
    if not workflow:
        workflow = []
    
    # Deduplicate workflow: remove exact duplicates
    seen = set()
    deduped_workflow = []
    for w in workflow:
        key = w.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped_workflow.append(w)
    workflow = deduped_workflow

    # Build rules section
    rules_block = "\n".join(f"- {r}" if not r.startswith("- ") else r for r in rules)
    
    # Build workflow section
    workflow_block = "\n".join(workflow)
    
    content = f"""---
name: agent-{name}
description: {desc}
category: development
---

# Agent {name.title()}

## Description

You are **{name.title()}**, {desc.lower()}.

## Rules

{rules_block}

## Workflow

{workflow_block}
"""
    return content


def convert_rule(file_path, metadata, body):
    name = file_path.stem.replace(".instructions", "").replace("-", " ").strip()
    desc = metadata.get("description", f"Rule: {name}")
    
    clean = clean_body(body)
    
    content = f"""---
name: rule-{name.lower().replace(" ", "-")}
description: {desc}
category: rules
---

# Rule: {name.title()}

## Description

This rule defines {desc.lower()}. It applies to the project context.

## Context

{clean}

## Application

When working on this project, always follow these guidelines.
"""
    return content


def convert_skill(file_path, metadata, body):
    name = metadata.get("name", file_path.parent.name)
    desc = metadata.get("description", "AIDD Skill")
    category = metadata.get("category", "development")
    
    clean = clean_body(body)
    if not clean.startswith("# "):
        clean = f"# Skill: {name}\n\n{clean}"
        
    content = f"""---
name: skill-aidd-{name.lower().replace(" ", "-")}
description: {desc}
category: {category}
---

# Skill: {name.title()}

## Description

{desc}

## Content

{clean}

## Instructions

Use this skill when performing tasks related to {name.lower()}.
"""
    return content


def convert_prompt(file_path, metadata, body):
    name = file_path.stem.replace(".prompt", "").replace("-", " ").strip()
    desc = metadata.get("description", f"AIDD Command: {name}")
    
    clean = clean_body(body)
    
    content = f"""---
name: prompt-aidd-{name.lower().replace(" ", "-")}
description: {desc}
category: commands
---

# Command: {name.title()}

## Description

{desc}

## Execution

Execute the following instructions:

```markdown
{clean}
```
"""
    return content


def convert_template(file_path, metadata, body):
    name = file_path.stem.replace("-", " ").strip()
    desc = metadata.get("description", f"Template: {name}")
    
    # Check if original body had frontmatter BEFORE clean_body stripped it
    template_meta = ""
    if body.startswith("---"):
        m = re.match(r"---\s*\n(.*?)\n---", body, re.DOTALL)
        if m:
            template_meta = m.group(0) + "\n\n"
    
    clean = clean_body(body)
    
    content = f"""---
name: template-{name.lower().replace(" ", "-")}
description: {desc}
category: templates
---

# Template: {name.title()}

## Description

Template for {desc.lower()}.

## Template

{template_meta}{clean}
"""
    return content


# --- Scanner & Orchestrator ---

def scan_and_convert(output_dir, types=None):
    results = []
    
    if types is None:
        types = ["agent", "rule", "skill", "prompt", "template"]
    
    # 1. Agents -> agents/<name>/SKILL.md
    if "agent" in types:
        agents_dir = AIDD_SRC / "agents"
        if agents_dir.exists():
            agents_out = output_dir / "agents"
            agents_out.mkdir(parents=True, exist_ok=True)
            for f in agents_dir.glob("*.agent.md"):
                fm, body = parse_frontmatter(f.read_text())
                md = convert_agent(f, fm, body)
                name = fm.get('name', f.stem)
                agent_dir = agents_out / name
                agent_dir.mkdir(parents=True, exist_ok=True)
                (agent_dir / "SKILL.md").write_text(md)
                results.append(f"agents/{name}")

    # 2. Rules -> rules/<name>/SKILL.md
    if "rule" in types:
        rules_dir = AIDD_SRC / "instructions"
        if rules_dir.exists():
            rules_out = output_dir / "rules"
            rules_out.mkdir(parents=True, exist_ok=True)
            for f in rules_dir.rglob("*.md"):
                if "instructions.md" in f.name or ".instructions" in f.stem:
                    fm, body = parse_frontmatter(f.read_text())
                    md = convert_rule(f, fm, body)
                    name = f.stem.replace(".instructions", "").replace("-", "-")
                    rule_dir = rules_out / name
                    rule_dir.mkdir(parents=True, exist_ok=True)
                    (rule_dir / "SKILL.md").write_text(md)
                    results.append(f"rules/{name}")

    # 3. Skills -> skills/<name>/SKILL.md
    if "skill" in types:
        skills_dir = AIDD_SRC / "skills"
        if skills_dir.exists():
            skills_out = output_dir / "skills"
            skills_out.mkdir(parents=True, exist_ok=True)
            for f in skills_dir.rglob("SKILL.md"):
                fm, body = parse_frontmatter(f.read_text())
                md = convert_skill(f, fm, body)
                name = f.parent.name
                skill_dir = skills_out / name
                skill_dir.mkdir(parents=True, exist_ok=True)
                (skill_dir / "SKILL.md").write_text(md)
                results.append(f"skills/{name}")

    # 4. Prompts -> prompts/<name>/SKILL.md
    if "prompt" in types:
        prompts_dir = AIDD_SRC / "prompts"
        if prompts_dir.exists():
            prompts_out = output_dir / "prompts"
            prompts_out.mkdir(parents=True, exist_ok=True)
            for f in prompts_dir.rglob("*.prompt.md"):
                fm, body = parse_frontmatter(f.read_text())
                md = convert_prompt(f, fm, body)
                name = f.stem.replace(".prompt", "")
                prompt_dir = prompts_out / name
                prompt_dir.mkdir(parents=True, exist_ok=True)
                (prompt_dir / "SKILL.md").write_text(md)
                results.append(f"prompts/{name}")

    # 5. Templates -> templates/<name>/SKILL.md
    if "template" in types:
        templates_dir = AIDD_DOCS / "templates"
        if templates_dir.exists():
            templates_out = output_dir / "templates"
            templates_out.mkdir(parents=True, exist_ok=True)
            for f in templates_dir.rglob("*.md"):
                fm, body = parse_frontmatter(f.read_text())
                md = convert_template(f, fm, body)
                name = f.stem
                template_dir = templates_out / name
                template_dir.mkdir(parents=True, exist_ok=True)
                (template_dir / "SKILL.md").write_text(md)
                results.append(f"templates/{name}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Bridge AIDD -> Hermes")
    parser.add_argument("--list", action="store_true", help="List detected elements")
    parser.add_argument("--install", action="store_true", help="Install to ~/.hermes/skills/")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    parser.add_argument("--types", nargs="+", 
                        choices=["agent", "rule", "skill", "prompt", "template"],
                        default=["agent", "rule", "skill", "prompt", "template"],
                        help="Which types to generate (default: all)")
    args = parser.parse_args()
    
    print(f"🔍 Scanning AIDD Framework in {AIDD_SRC}...")
    
    # Ensure output dir
    HERMES_SKILLS.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        print("Detected elements:")
        if (AIDD_SRC / "agents").exists():
            agents = list((AIDD_SRC / 'agents').glob('*.agent.md'))
            print(f"  - Agents: {len(agents)}")
        if (AIDD_SRC / "instructions").exists():
            rules = list((AIDD_SRC / 'instructions').rglob('*.md'))
            rules = [r for r in rules if "instructions.md" in r.name or ".instructions" in r.stem]
            print(f"  - Rules: {len(rules)}")
        if (AIDD_SRC / "skills").exists():
            skills_list = list((AIDD_SRC / "skills").rglob("SKILL.md"))
            print(f"  - Skills: {len(skills_list)}")
        if (AIDD_SRC / "prompts").exists():
            prompts = list((AIDD_SRC / 'prompts').rglob('*.prompt.md'))
            print(f"  - Prompts: {len(prompts)}")
        if (AIDD_DOCS / "templates").exists():
            templates = list((AIDD_DOCS / 'templates').rglob('*.md'))
            print(f"  - Templates: {len(templates)}")
        return

    # Convert
    generated = scan_and_convert(HERMES_SKILLS, types=args.types)
    
    print(f"✅ {len(generated)} elements converted to Hermes SKILL.md format in {HERMES_SKILLS}")
    
    # Group by type
    by_type = {}
    for g in generated:
        t = g.split("-")[0]
        by_type.setdefault(t, []).append(g)
    for t, items in by_type.items():
        print(f"  - {t}: {len(items)}")
    
    if not generated:
        print("⚠️  No elements found to convert. Check source directory structure.")
        return

    if args.install:
        print("\n🚀 Installing to ~/.hermes/skills/aidd-overlay/ ...")
        HERMES_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Move files
        for f in HERMES_SKILLS.glob("*.md"):
            shutil.copy2(f, HERMES_INSTALL_DIR / f.name)
        
        # Create a README for the collection
        import datetime
        readme = f"""# AIDD Overlay Skills

Generated on {datetime.date.today().isoformat()}

## Available Skills
"""
        for name in generated:
            readme += f"- {name}\n"
            
        (HERMES_INSTALL_DIR / "README.md").write_text(readme)
        print(f"✅ Installed {len(generated)} skills in {HERMES_INSTALL_DIR}")
    else:
        print("\n💡 To install these skills permanently, run:")
        print(f"   python3 scripts/aidd-to-hermes.py --install")


if __name__ == "__main__":
    main()
