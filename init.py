#!/usr/bin/env python3
"""
init - Initialiser la mémoire interne AIDD

Cette commande crée et initialise les fichiers de mémoire du projet.
Usage: python init.py
"""

import os
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
AIDD_DIR = os.path.join(PROJECT_ROOT, "aidd_docs")

def create_memory_files():
    """Créer les fichiers de mémoire AIDD"""
    
    memory_dir = os.path.join(AIDD_DIR, "memory")
    os.makedirs(memory_dir, exist_ok=True)
    
    # 1. DECISIONS.md - Décisions architecturales
    decisions_file = os.path.join(memory_dir, "DECISIONS.md")
    if not os.path.exists(decisions_file):
        with open(decisions_file, "w", encoding="utf-8") as f:
            f.write("""# 🧠 Décisions Architecturales

## Structure du projet
- **aidd_docs/**: Documentation AIDD (plans, reviews, memory, changelog)
- **scripts/**: Scripts Python (scraping, training, conversion)
- **data/**: Données (raw, clean, final)
- **logs/**: Logs d'exécution
- **config/**: Configuration (Axolotl, scraping, etc.)

## Technologies choisies
- **Scraping**: Python, requests, BeautifulSoup4
- **Training**: Axolotl, PyTorch, LoRA/QLoRA
- **API**: Together.ai, Fireworks.ai, Hugging Face
- **Versioning**: Git, GitHub CLI
- **Workflow**: AIDD (tasks, challenge, plans, learn)

## Convention de code
- **Naming**: snake_case pour Python, kebab-case pour fichiers
- **Documentation**: Docstrings, commentaires explicatifs
- **Testing**: Tests unitaires avant implémentation (TDD)
- **Git**: Branches dédiées, PRs obligatoires, merge protégé

## Sécurité
- **Tokens**: `.env` (non versionné)
- **API Keys**: Variables d'environnement Railway
- **GitHub**: Fine-grained tokens avec permissions minimales
""")
        print("✅ DECISIONS.md créé")
    
    # 2. LESSONS.md - Lessons learned global
    lessons_file = os.path.join(memory_dir, "LESSONS.md")
    if not os.path.exists(lessons_file):
        with open(lessons_file, "w", encoding="utf-8") as f:
            f.write("""# 📚 Lessons Learned - Projet Suddenly AI Hub

## 📝 Historique des apprentissages

### Session 1 - Exploration jdRoll
- **Succès**: Script de scraping fonctionnel
- **Échec**: Regex complexes non supportées
- **Leçon**: Utiliser des regex simples et testées

### Session 2 - Scraping 20 campagnes
- **Succès**: Structure de données JSON
- **Échec**: Pas de logs générés
- **Leçon**: Toujours ajouter `try/except` partout

### Phase 3 - Fine-tuning
- **Succès**: Scripts prêts
- **Échec**: Données non disponibles
- **Leçon**: Attendre données complètes avant de commencer

---

## 📋 Méthodologie AIDD

### Planning
- **Writing-plans**: Documentation structurée avant implémentation
- **Challenge-plan**: Validation systématique (9+/10 requis)
- **TDD**: Tests avant implémentation
- **Learn**: Documentation immédiate après chaque phase

### Workflow Git
- **Branches**: Une branche par phase/feature
- **PRs**: Obligatoires pour merge
- **Commits**: Mesgés conventionnels
- **Merge**: Fast-forward uniquement

---

## 💡 Bonnes pratiques

1. **Documentation**: Écrire avant de coder
2. **Validation**: Challenges systématiques
3. **Testing**: Tests unitaires obligatoires
4. **Security**: Tokens dans variables d'environnement
5. **Git**: Branches protégées, PRs reviewées

---

**Dernière mise à jour**: 2026-05-13
**Auteur**: RebelliousSmile
""")
        print("✅ LESSONS.md créé")
    
    # 3. PROJECT_INFO.json - Informations projet
    info_file = os.path.join(memory_dir, "PROJECT_INFO.json")
    if not os.path.exists(info_file):
        project_info = {
            "project_name": "Suddenly AI Hub",
            "description": "Fine-tuning LLM pour JDR français",
            "version": "1.0.0",
            "start_date": "2026-05-13",
            "author": "RebelliousSmile",
            "license": "MIT",
            "workflow": "AIDD",
            "phases": {
                "session1": {"status": "completed", "description": "Exploration jdRoll"},
                "session2": {"status": "pending", "description": "Scraping 20 campagnes"},
                "phase3": {"status": "planned", "description": "Fine-tuning Qwen2.5-7B"},
                "phase4": {"status": "planned", "description": "Amélioration continue"},
                "phase5": {"status": "planned", "description": "Scaling 100+ campagnes"},
                "phase6": {"status": "planned", "description": "Production & Haute disponibilité"},
                "phase7": {"status": "planned", "description": "Communauté & Open source"}
            },
            "tech_stack": {
                "scraping": ["Python", "requests", "beautifulsoup4"],
                "training": ["Axolotl", "PyTorch", "LoRA", "QLoRA"],
                "inference": ["Together.ai", "Fireworks.ai", "vLLM"],
                "storage": ["Hugging Face", "GitHub"],
                "workflow": ["AIDD", "GitHub CLI", "Git"]
            },
            "api_keys": {
                "together_ai": "configured",
                "fireworks_ai": "configured",
                "github": "configured",
                "hugging_face": "configured"
            },
            "metrics": {
                "total_scripts": 13,
                "total_plans": 7,
                "total_reviews": 7,
                "average_review_score": 9.5
            }
        }
        
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(project_info, f, indent=2, ensure_ascii=False)
        print("✅ PROJECT_INFO.json créé")
    
    # 4. WORKFLOW.md - Guide workflow
    workflow_file = os.path.join(PROJECT_ROOT, "aidd_docs", "WORKFLOW.md")
    if not os.path.exists(workflow_file):
        with open(workflow_file, "w", encoding="utf-8") as f:
            f.write("""# 🔄 Workflow AIDD - Suddenly AI Hub

## 📋 Structure

```
aidd_docs/
├── tasks/           # Plans AIDD (issues)
├── plans/           # Plans détaillés
├── reviews/         # Challenges et validations
├── memory/          # Mémoire interne (LESSONS, DECISIONS)
└── changelog/       # Changelog et versions
```

## 🎯 Processus

### 1. Planification (Issue → Tasks)
- Créer issue GitHub
- Écrire plan dans `aidd_docs/tasks/`
- Utiliser `writing-plans` skill

### 2. Validation (Challenge)
- Lancer `challenge-plan` skill
- Score ≥ 9/10 requis
- Documenter dans `aidd_docs/reviews/`

### 3. Implémentation (Git)
- Créer branche dédiée
- Implémenter selon plan
- Tests TDD obligatoires

### 4. Review (PR)
- Créer Pull Request
- Code review obligatoire
- Merge après approbation

### 5. Documentation (Learn)
- Documenter lessons learned
- Mettre à jour `LESSONS.md`
- Changelog à jour

## 🛠️ Skills AIDD

- `writing-plans`: Documentation structurée
- `challenge-plan`: Validation automatisée
- `test-driven-development`: Tests avant implémentation
- `learn`: Documentation des apprentissages
- `monitoring`: Suivi des métriques

## 📝 Templates

### Plan de tâche
```yaml
---
issue_id: #50
title: Phase X - Description
author: RebelliousSmile
created_at: YYYY-MM-DD
status: planned
---
```

### Rapport de challenge
```yaml
---
session: X
plan_ref: phaseX_*.md
score: X.X/10
status: approved/rejected
---
```

---

**Dernière mise à jour**: 2026-05-13
**Mainteneur**: RebelliousSmile
""")
        print("✅ WORKFLOW.md mis à jour")
    
    print("\n✅ Tous les fichiers de mémoire créés !")

def main():
    """Point d'entrée principal"""
    
    print("="*80)
    print("🚀 INIT AIDD - Initialisation de la mémoire interne")
    print("="*80)
    print(f"Project: {PROJECT_ROOT}")
    print(f"AIDD dir: {AIDD_DIR}")
    print("="*80)
    
    # Créer les fichiers
    create_memory_files()
    
    # Afficher la structure
    print("\n📁 Structure aidd_docs/:")
    for root, dirs, files in os.walk(AIDD_DIR):
        level = root.replace(AIDD_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    print("\n" + "="*80)
    print("✅ INITIATION TERMINÉE")
    print("="*80)
    print("\nProchaines étapes:")
    print("1. Créer une nouvelle issue GitHub")
    print("2. Écrire un plan dans aidd_docs/tasks/")
    print("3. Lancer challenge-plan")
    print("4. Implémenter et créer PR")
    print("\n🎯 Workflow AIDD est prêt !")
    print("="*80)

if __name__ == "__main__":
    main()
