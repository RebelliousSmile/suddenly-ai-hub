# AIDD Documentation - Workflow Standard

## 🎯 Workflow d'une Feature

### 1. Brainstorm
- Discussion informelle sur la feature
- Identification des contraintes et objectifs
- Détermination du scope

### 2. Issue GitHub
- Créer une issue GitHub avec le résumé
- Ajouter les labels appropriés
- Assigner à la milestone correspondante

### 3. Plan AIDD
- Créer le plan dans `aidd_docs/tasks/<task_id>.md`
- Décomposer en sous-tâches
- Définir les critères d'acceptation
- Utiliser le template `TASK_TEMPLATE.md`

### 4. Challenge
- Utiliser la skill `challenge` pour valider le plan
- Résoudre tous les "deal breakers" avant d'implémenter
- Ajuster le plan si nécessaire

### 5. TDD (Test-Driven Development)
- Écrire les tests d'abord (RED)
- Implémenter le code minimum pour passer les tests (GREEN)
- Refactoriser le code

### 6. Implémentation
- Développer la feature
- Suivre les conventions de code
- Ajouter la documentation nécessaire

### 7. Revue Fonctionnelle
- Tester la feature en situation réelle
- Vérifier les critères d'acceptation
- Collecter les feedbacks

### 8. Revue de Code
- Review du code par un pair
- Vérifier la qualité et les conventions
- S'assurer que les tests passent

### 9. Commit & Push
- Commit avec message conventionnel
- Push vers la branche
- Créer une Pull Request

### 10. Merge & Deploy
- Merge dans la branche principale
- Déployer en environnement de test
- Vérifier que tout fonctionne

### 11. End Plan
- Exécuter `end_plan` pour fermer les branches
- Mettre à jour les tâches dans `aidd_docs/tasks/`
- Marquer l'issue GitHub comme résolue

### 12. Learn & Changelog
- Documenter les apprentissages dans `aidd_docs/memory/`
- Mettre à jour le changelog
- Gestion des versions (semantic versioning)
- Noter les améliorations possibles

## 📁 Structure des fichiers

```
aidd_docs/
├── tasks/              # Plans de tâches par issue/task
│   ├── TASK_TEMPLATE.md
│   └── <task_id>.md
├── memory/             # Connaissances accumulées
│   ├── DECISIONS.md    # Décisions architecturales
│   └── LESSONS.md      # Leçons apprises
├── plans/              # Plans détaillés
│   └── <project>_plan.md
├── reviews/            # Revues de code et fonctionnelles
│   └── <feature>_review.md
└── changelog/          # Changelog et versions
    ├── CHANGELOG.md
    └── VERSIONS.md
```

## 📝 Templates

### TASK_TEMPLATE.md
Voir `aidd_docs/tasks/TASK_TEMPLATE.md`

### DECISIONS.md
Voir `aidd_docs/memory/DECISIONS.md`

### CHANGELOG.md
Voir `aidd_docs/changelog/CHANGELOG.md`

## 🚀 Scripts utilitaires

### `scripts/end_plan.sh`
Ferme les branches et met à jour le tracker de tâches.

### `scripts/learn.sh`
Documente les apprentissages et met à jour le changelog.

### `scripts/changelog.sh`
Gère les mises à jour du changelog et la version.

## ⚠️ Règles importantes

1. **Jamais de code avant le plan** - Toujours créer le plan et le challenge d'abord
2. **TDD quand possible** - Écrire les tests avant le code
3. **Review obligatoire** - Aucun merge sans revue de code
4. **Documentation à jour** - Mettre à jour la doc en même temps que le code
5. **Commits atomiques** - Un commit = une fonctionnalité / un bugfix
6. **Changelog précis** - Documenter chaque changement important

## 📚 Ressources

- [GitHub Issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)

---

**Dernière mise à jour:** 2026-05-13
**Auteur:** RebelliousSmile
