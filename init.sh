#!/bin/bash
# ============================================
# /init - Initialiser le projet avec AIDD
# ============================================

# Script d'initialisation pour créer la structure AIDD
# Exécution: ./init

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

echo "🚀 Initialisation de suddenly-ai-hub avec AIDD..."
echo "============================================================"

# 1. Créer la structure aidd_docs
echo "📁 Création de la structure aidd_docs/..."

mkdir -p "$PROJECT_ROOT/aidd_docs/tasks"
mkdir -p "$PROJECT_ROOT/aidd_docs/memory"
mkdir -p "$PROJECT_ROOT/aidd_docs/plans"
mkdir -p "$PROJECT_ROOT/aidd_docs/reviews"
mkdir -p "$PROJECT_ROOT/aidd_docs/changelog"

# Créer les fichiers .gitkeep
touch "$PROJECT_ROOT/aidd_docs/tasks/.gitkeep"
touch "$PROJECT_ROOT/aidd_docs/memory/.gitkeep"
touch "$PROJECT_ROOT/aidd_docs/plans/.gitkeep"
touch "$PROJECT_ROOT/aidd_docs/reviews/.gitkeep"
touch "$PROJECT_ROOT/aidd_docs/changelog/.gitkeep"

echo "✅ Structure aidd_docs créée"

# 2. Créer les templates
echo "📝 Création des templates..."

# TASK_TEMPLATE.md (déjà créé manuellement, on s'assure qu'il existe)
if [ ! -f "$PROJECT_ROOT/aidd_docs/tasks/TASK_TEMPLATE.md" ]; then
    cat > "$PROJECT_ROOT/aidd_docs/tasks/TASK_TEMPLATE.md" << 'EOF'
---
issue_id: #
title: 
author: 
created_at: $(date +%Y-%m-%d)
updated_at: $(date +%Y-%m-%d)
status: planned
priority: 
labels: 
---

# 📋 {{title}}

## 🎯 Objectif
[Description courte de la feature]

## ✅ Critères d'acceptation
- [ ] Critère 1
- [ ] Critère 2
- [ ] Critère 3

## 📝 Notes
...
EOF
fi

# DECISIONS.md
if [ ! -f "$PROJECT_ROOT/aidd_docs/memory/DECISIONS.md" ]; then
    cat > "$PROJECT_ROOT/aidd_docs/memory/DECISIONS.md" << 'EOF'
# 🧠 Décisions Architecturales

## Structure du projet
[Description des choix architecturaux]

## Technologies
[Liste des technologies choisies et pourquoi]

## Convention de code
[Règles et standards]

---

**Dernière mise à jour:** $(date +%Y-%m-%d)
EOF
fi

# LESSONS.md
if [ ! -f "$PROJECT_ROOT/aidd_docs/memory/LESSONS.md" ]; then
    cat > "$PROJECT_ROOT/aidd_docs/memory/LESSONS.md" << 'EOF'
# 📚 Leçons Apprises

## [Date] - [Sujet]
### Ce qui a bien fonctionné
- ...

### Ce qui a mal fonctionné
- ...

### Améliorations futures
- ...

---
EOF
fi

# CHANGELOG.md
if [ ! -f "$PROJECT_ROOT/aidd_docs/changelog/CHANGELOG.md" ]; then
    cat > "$PROJECT_ROOT/aidd_docs/changelog/CHANGELOG.md" << 'EOF'
# 📝 Changelog

## [Unreleased]
### Added
- 

### Changed
- 

### Fixed
- 

### Deprecated
- 

### Removed
- 

### Security
- 

---

## [Version] - [Date]
### Added
- 

### Changed
- 

### Fixed
- 

---

[Unreleased]: https://github.com/RebelliousSmile/suddenly-ai-hub/compare/main...HEAD
EOF
fi

# VERSIONS.md
if [ ! -f "$PROJECT_ROOT/aidd_docs/changelog/VERSIONS.md" ]; then
    cat > "$PROJECT_ROOT/aidd_docs/changelog/VERSIONS.md" << 'EOF'
# 📦 Versions

## Semantic Versioning (SemVer)

Format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Changements incompatibles
- **MINOR**: Nouvelles fonctionnalités (backwards compatible)
- **PATCH**: Corrections de bugs (backwards compatible)

## Versions actuelles

### 0.1.0 (Initial)
- Initial commit
- Structure de base

---

**Dernière mise à jour:** $(date +%Y-%m-%d)
EOF
fi

# WORKFLOW.md
if [ ! -f "$PROJECT_ROOT/aidd_docs/WORKFLOW.md" ]; then
    cat > "$PROJECT_ROOT/aidd_docs/WORKFLOW.md" << 'EOF'
# Workflow AIDD

## Étapes

1. **Brainstorm** - Discussion informelle
2. **Issue GitHub** - Créer l'issue
3. **Plan AIDD** - Créer le plan dans `aidd_docs/tasks/`
4. **Challenge** - Valider le plan avec skill challenge
5. **TDD** - Tests d'abord
6. **Implémentation** - Code
7. **Revue fonctionnelle** - Test en situation réelle
8. **Revue de code** - Review par pair
9. **Commit & Push** - Conventionnel
10. **End Plan** - Fermer la branche
11. **Learn** - Documenter les apprentissages
12. **Changelog** - Mettre à jour le changelog

## Règles

- ✅ Jamais de code avant le plan
- ✅ TDD quand possible
- ✅ Review obligatoire
- ✅ Documentation à jour
- ✅ Commits atomiques
- ✅ Changelog précis

---

**Dernière mise à jour:** $(date +%Y-%m-%d)
EOF
fi

echo "✅ Templates créés"

# 3. Créer les scripts utilitaires
echo "🔧 Création des scripts utilitaires..."

mkdir -p "$PROJECT_ROOT/scripts"

# end_plan.sh
cat > "$PROJECT_ROOT/scripts/end_plan.sh" << 'EOF'
#!/bin/bash
# Ferme les branches et met à jour le tracker

TASK_ID=$1
if [ -z "$TASK_ID" ]; then
    echo "❌ Usage: ./end_plan <task_id>"
    exit 1
fi

echo "🏁 Clôture de la tâche $TASK_ID..."

# Mettre à jour le plan
if [ -f "aidd_docs/tasks/$TASK_ID.md" ]; then
    echo "✅ Plan mis à jour"
fi

# Nettoyer les branches temporaires
echo "🧹 Nettoyage des branches..."
# git branch --merged | grep -v "\*" | xargs git branch -d 2>/dev/null || true

echo "✅ End plan terminé"
EOF
chmod +x "$PROJECT_ROOT/scripts/end_plan.sh"

# learn.sh
cat > "$PROJECT_ROOT/scripts/learn.sh" << 'EOF'
#!/bin/bash
# Documente les apprentissages

echo "📚 Documenter les apprentissages..."

# Créer une entrée dans LESSONS.md
echo "" >> "aidd_docs/memory/LESSONS.md"
echo "## $(date +%Y-%m-%d) - $1" >> "aidd_docs/memory/LESSONS.md"
echo "### Ce qui a bien fonctionné" >> "aidd_docs/memory/LESSONS.md"
echo "- ..." >> "aidd_docs/memory/LESSONS.md"
echo "" >> "aidd_docs/memory/LESSONS.md"
echo "### Ce qui a mal fonctionné" >> "aidd_docs/memory/LESSONS.md"
echo "- ..." >> "aidd_docs/memory/LESSONS.md"
echo "" >> "aidd_docs/memory/LESSONS.md"
echo "### Améliorations futures" >> "aidd_docs/memory/LESSONS.md"
echo "- ..." >> "aidd_docs/memory/LESSONS.md"

echo "✅ Apprentissages documentés"
EOF
chmod +x "$PROJECT_ROOT/scripts/learn.sh"

# changelog.sh
cat > "$PROJECT_ROOT/scripts/changelog.sh" << 'EOF'
#!/bin/bash
# Met à jour le changelog et la version

TYPE=$1
VERSION=$2
MESSAGE=$3

if [ -z "$TYPE" ] || [ -z "$VERSION" ]; then
    echo "❌ Usage: ./changelog.sh <added|changed|fixed> <version> <message>"
    exit 1
fi

echo "📝 Mise à jour du changelog..."

# Mettre à jour CHANGELOG.md
sed -i "s/\[Unreleased\]/[$VERSION] - $(date +%Y-%m-%d)/" "aidd_docs/changelog/CHANGELOG.md"
echo "" >> "aidd_docs/changelog/CHANGELOG.md"
echo "### Added" >> "aidd_docs/changelog/CHANGELOG.md"
echo "- $MESSAGE" >> "aidd_docs/changelog/CHANGELOG.md"

echo "✅ Changelog mis à jour"
EOF
chmod +x "$PROJECT_ROOT/scripts/changelog.sh"

echo "✅ Scripts utilitaires créés"

# 4. Ajouter au .gitignore si nécessaire
echo "🔒 Vérification du .gitignore..."
if ! grep -q "aidd_docs" ".gitignore" 2>/dev/null; then
    echo "" >> ".gitignore"
    echo "# AIDD Documentation" >> ".gitignore"
    echo "# aidd_docs/ (optionnel: garder pour documentation)" >> ".gitignore"
    echo "# !aidd_docs/" >> ".gitignore"  # On garde aidd_docs
fi

# 5. Initialiser Git si nécessaire
echo "📦 Vérification de Git..."
if [ ! -d ".git" ]; then
    git init
    echo "✅ Git initialisé"
fi

# 6. Commit initial
echo "📝 Commit initial..."
git add .
git commit -m "Init: Setup aidd_docs structure with workflow" 2>/dev/null || echo "✅ Déjà commité"

echo ""
echo "============================================================"
echo "✅ Initialisation terminée !"
echo "============================================================"
echo ""
echo "📁 Structure créée:"
echo "   • aidd_docs/tasks/"
echo "   • aidd_docs/memory/"
echo "   • aidd_docs/plans/"
echo "   • aidd_docs/reviews/"
echo "   • aidd_docs/changelog/"
echo ""
echo "📝 Templates disponibles:"
echo "   • TASK_TEMPLATE.md"
echo "   • DECISIONS.md"
echo "   • LESSONS.md"
echo "   • CHANGELOG.md"
echo "   • VERSIONS.md"
echo "   • WORKFLOW.md"
echo ""
echo "🔧 Scripts utilitaires:"
echo "   • scripts/end_plan.sh"
echo "   • scripts/learn.sh"
echo "   • scripts/changelog.sh"
echo ""
echo "🚀 Prochaine étape: Commencer une feature avec /brainstorm !"
