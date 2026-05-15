# Audit Bridge AIDD -> Hermes - Comparatif sémantique

**Date:** 2026-05-14
**Méthode:** Analyse manuelle des contenus réels (pas de mots-clés)

## 1. Agents vs Skills de workflow

| Agent AIDD | Fonction | Équivalent Hermes le + proche | Chevauchement |
|---|---|---|---|
| **Kent** | Planification & Architecture | `writing-plans` + `plan` | 🟡 PARTIEL — Kent est un AGENT (persona à déléguer), writing-plans est une SKILL procédurale. Non redondant. |
| **Alexia** | Implémentation autonome | `subagent-driven-development` | 🟡 PARTIEL — Le skill Hermès fait 2-stage review (spec + quality), Alexia fait l'auto-correction. Complémentaire. |
| **Claire** | Documentation & Tech Writer | `powerpoint`, `ocr-and-documents` | 🔵 UNIQUE — Aucun skill doc technique. |
| **Iris** | Full-Stack Dev | `aidd-dev-workflow` | 🟡 PARTIEL — Iris est un agent, le workflow est une skill gateway. |
| **Martin** | Deployment/DevOps | `devops/webhook-subscriptions` | 🔵 UNIQUE — Aucun skill deployment spécifique. |

## 2. Skills/Workflows AIDD vs Skills Hermes

### AIDD `aidd-auto-implement` (le workflow complet)
```
brainstorm → plan → implement → test → review → PR
```

| Étape | Prompt AIDD | Skill Hermes équivalente | Match? |
|---|---|---|---|
| Brainstorm | `02-brainstorm.prompt.md` | *(rien de spécifique)* | 🔵 UNIQUE |
| Planifier | `03-plan.prompt.md` | `writing-plans` | 🔴 **DOUBLON** — Même chose : plan détaillé avec tâches bite-sized, fichiers exacts, code complet. |
| Implémenter | `04-implement.prompt.md` | `subagent-driven-development` | 🔴 **DOUBLON** — Même workflow : delegate_task par tâche, 2-stage review. |
| Tester | `06-test.prompt.md` | `test-driven-development` | 🟡 CHEVAUCHEMENT — TDD Hermès est plus strict (iron law, anti-patterns). Le prompt AIDD est un workflow de test, pas uniquement TDD. |
| Review | `05-review-code.prompt.md` | `requesting-code-review` | 🔴 **DOUBLON** — Même pipeline : static scan + reviewer subagent + auto-fix. |
| PR | `08-create-request.prompt.md` | *(rien de spécifique)* | 🔵 UNIQUE |

### AIDD `aidd-challenge` (review critique)
| Fonction | Prompt AIDD | Skill Hermes | Match? |
|---|---|---|---|
| Review critique | `02-challenge.prompt.md` | `requesting-code-review` | 🟡 CHEVAUCHEMENT — Le challenge est un review critique (très proche). Mais "challenge" est plus large (peut challenger le design, pas que le code). |

## 3. Les 49 prompts AIDD — Analyses détaillées

### Phase : PLANIFICATION
| Prompt | Description | Hermes équivalent | Verdict |
|---|---|---|---|
| `01-init` | Créer/mettre à jour la memory bank | `aidd-workspace` | 🟡 PARTIEL — AIDD workspace existe déjà. |
| `01-onboard` | Détecter l'état du projet | *(rien)* | 🔵 UNIQUE |
| `02-brainstorm` | Brainstorming interactif | *(rien)* | 🔵 UNIQUE |
| `03-plan` | Générer un plan technique | `writing-plans` | 🔴 **DOUBLON** |
| `10-new-issue` | Créer une issue GitHub | `github-issues` | 🟡 PARTIEL — L'issue Hermès est plus complète (triage, labels). |
| `10-reflect-issue` | Réfléchir à une issue | *(rien)* | 🔵 UNIQUE |
| `02-ticket-info` | Info sur un ticket | *(rien)* | 🔵 UNIQUE |

### Phase : IMPLÉMENTATION
| Prompt | Description | Hermes équivalent | Verdict |
|---|---|---|---|
| `04-implement` | Implémenter un plan | `subagent-driven-development` | 🔴 **DOUBLON** |
| `04-implement-from-design` | Implémenter depuis un design Figma | *(rien)* | 🔵 UNIQUE |
| `01-generate-agent` | Créer un agent | `aidd-workspace` | 🟡 PARTIEL — Workspace peut faire ça. |
| `01-generate-skill` | Créer un skill | `aidd-workspace` | 🟡 PARTIEL |
| `01-generate-rules` | Créer des rules | `aidd-workspace` | 🟡 PARTIEL |
| `01-generate-command` | Créer une commande | `aidd-workspace` | 🟡 PARTIEL |
| `01-generate-architecture` | Générer une architecture | `architecture-diagram` | 🟡 PARTIEL — Architecture-diagram fait du SVG, pas du Mermaid. |
| `04-run-projection` | Exécuter une projection | *(rien)* | 🔵 UNIQUE |

### Phase : TESTING
| Prompt | Description | Hermes équivalent | Verdict |
|---|---|---|---|
| `06-test` | Lister non-testé + créer tests | `test-driven-development` | 🟡 CHEVAUCHEMENT — Test-driven est TDD strict (test avant code), ce prompt est plus général (créer des tests pour du code existant). |
| `06-test-journey` | Tests de parcours utilisateur | *(rien)* | 🔵 UNIQUE |
| `test_bruno` | Tests Bruno (API) | *(rien)* | 🔵 UNIQUE |

### Phase : REVIEW
| Prompt | Description | Hermes équivalent | Verdict |
|---|---|---|---|
| `05-review-code` | Review code quality | `requesting-code-review` | 🔴 **DOUBLON** — Static scan + reviewer subagent + auto-fix. |
| `05-review-functional` | Review fonctionnelle | `requesting-code-review` | 🟡 CHEVAUCHEMENT — Le code-review Hermès inclut un review fonctionnel (Step 2: static scan). |
| `02-challenge` | Critique/Challenge | `requesting-code-review` | 🟡 CHEVAUCHEMENT — Proche mais le challenge est plus large. |
| `09-audit` | Analyse tech debt | `requesting-code-review` | 🟡 PARTIEL — L'audit est plus large (tech debt, pas juste security). |
| `09-security-refactor` | Sécurisation | `requesting-code-review` | 🟡 CHEVAUCHEMENT — Le security scan existe dans requesting-code-review. |

### Phase : DEVOPS/OTHER
| Prompt | Description | Hermes équivalent | Verdict |
|---|---|---|---|
| `08-commit` | Git commit | *(rien de spécifique)* | 🔵 UNIQUE |
| `08-create-request` | Créer PR/MR | `github-issues` (PR management) | 🟡 PARTIEL — github-issues gère les issues, pas vraiment les PRs. |
| `08-tag` | Git tag | *(rien)* | 🔵 UNIQUE |
| `07-mermaid` | Diagrammes Mermaid | `architecture-diagram` | 🟢 UNIQUE — Architecture-diagram fait du SVG, pas du Mermaid. |
| `07-learn` | Apprentissage/analyse | `aidd-workspace` | 🟡 PARTIEL |
| `10-debug` | Debugging | `debugging-hermes-tui` ? | 🟡 CHEVAUCHEMENT — À vérifier. |
| `10-reproduce` | Reproduire un bug | *(rien)* | 🔵 UNIQUE |
| `decompose_mikado` | Méthode Mikado | *(rien)* | 🔵 UNIQUE |
| `previously` | Historique de changements | `aidd-workspace` | 🟡 PARTIEL |
| `journey` | Parcours utilisateur | *(rien)* | 🔵 UNIQUE |
| `taste` | Analyse codebase | `aidd-workspace` | 🟡 PARTIEL |
| `end_plan` | Fin de plan | *(rien)* | 🔵 UNIQUE |
| `foresee` | Prédiction/Risque | *(rien)* | 🔵 UNIQUE |
| `project_memory` | Mémoire projet | `aidd-workspace` | 🟡 PARTIEL |
| `project_status` | Status projet | `aidd-workspace` | 🟡 PARTIEL |
| `changelog` | Changelog | *(rien)* | 🔵 UNIQUE |
| `audit_memory` | Audit mémoire | `aidd-workspace` | 🟡 PARTIEL |
| `09-performance` | Optimisation perf | *(rien)* | 🔵 UNIQUE |
| `04-assert` | Assertions/Spécifications | *(rien)* | 🔵 UNIQUE |
| `04-assert-architecture` | Assertions architecture | *(rien)* | 🔵 UNIQUE |
| `04-assert-frontend` | Assertions frontend | *(rien)* | 🔵 UNIQUE |
| `03-image-extract-details` | Extraire détails d'image | *(rien)* | 🔵 UNIQUE |
| `03-components-behavior` | Composants/behavior | *(rien)* | 🔵 UNIQUE |
| `02-create-user-stories` | User stories | *(rien)* | 🔵 UNIQUE |
| `01-init` | Memory bank | `aidd-workspace` | 🟡 PARTIEL |

## 4. Synthèse des chevauchements

### 🔴 DOUBLONS COMPLETS (même fonctionnalité, même workflow)
| AIDD | Hermes | Décision |
|---|---|---|
| `03-plan.prompt.md` | `writing-plans` | **SUPPRIMER** le prompt AIDD — Hermès est plus complet |
| `04-implement.prompt.md` | `subagent-driven-development` | **SUPPRIMER** le prompt AIDD — Hermès est plus complet |
| `05-review-code.prompt.md` | `requesting-code-review` | **SUPPRIMER** le prompt AIDD — Hermès est plus complet |

### 🟡 CHEVAUCHEMENTS PARTIELS (fonctionnalités proches mais différentes)
| AIDD | Hermes | Décision |
|---|---|---|
| `06-test.prompt.md` | `test-driven-development` | **GARDER** — Test est plus général (créer tests sur code existant) |
| `05-review-functional.prompt.md` | `requesting-code-review` | **GARDER** — Review fonctionnelle vs code quality |
| `02-challenge.prompt.md` | `requesting-code-review` | **GARDER** — Challenge plus large (design, pas que code) |
| `09-audit.prompt.md` | `requesting-code-review` | **GARDER** — Audit tech debt vs security |
| `09-security-refactor.prompt.md` | `requesting-code-review` | **GARDER** — Security refactor spécifique |

### 🔵 UNIQUES (aucun équivalent)
- `02-brainstorm` → UNIQUE
- `04-implement-from-design` → UNIQUE
- `08-commit` → UNIQUE
- `08-create-request` → UNIQUE (github-issues fait les issues, pas les PRs)
- `07-mermaid` → UNIQUE (SVG vs Mermaid)
- `01-onboard` → UNIQUE
- `10-debug` → À vérifier
- `10-reproduce` → UNIQUE
- `decompose_mikado` → UNIQUE
- `journey` → UNIQUE
- `09-performance` → UNIQUE
- `04-assert` → UNIQUE
- `02-create-user-stories` → UNIQUE
- `changelog` → UNIQUE
- **Et ~30 autres prompts spécifiques**

## 5. Conclusion

Sur 49 prompts AIDD :
- **3 doublons complets** à supprimer (`03-plan`, `04-implement`, `05-review-code`)
- **5 chevauchements partiels** à garder (`06-test`, `05-review-functional`, `02-challenge`, `09-audit`, `09-security-refactor`)
- **~41 uniques** à conserver

**Gain net : ~41 prompts uniques + 5 skills/rules agents.**
