# Intégration aidd-framework — suddenly-muses

Ce projet utilise **aidd-custom** et **aidd-overlay** pour structurer ses règles, agents, skills et commandes selon les conventions d'aidd-framework.

## Composants externes

| Composant | Rôle | Repo |
|---|---|---|
| `aidd-custom` | CLI de bootstrapping / mise à jour des conventions aidd dans un projet | `RebelliousSmile/aidd-custom` |
| `aidd-overlay` | Surcharges et templates spécifiques à l'organisation | `RebelliousSmile/aidd-overlay` |

L'installation pratique est décrite dans `INSTALL.md`.

## Mapping conventions → emplacements

| Type | Emplacement local | Référence |
|---|---|---|
| Agents | `.claude/agents/` (flat) | `.claude/rules/04-tooling/ide-mapping.md` |
| Commands | `.claude/commands/<phase>/` (phase préfixée `01_` à `10_`) | `.claude/rules/01-standards/1-command-structure.md` |
| Rules | `.claude/rules/<category>/` | idem |
| Skills | `.claude/skills/<skill-name>/SKILL.md` | idem |
| Context principal | `CLAUDE.md` à la racine | idem |
| Mémoire projet | `aidd_docs/memory/*.md` | référencée depuis `CLAUDE.md` |
| Mémoire externe | `aidd_docs/memory/external/*.md` | chargée à la demande |
| Mémoire interne | `aidd_docs/memory/internal/*.md` | chargée à la discrétion de l'agent |

## Taxonomie SDLC des commandes

Phases utilisées comme préfixes de sous-dossiers dans `.claude/commands/` :

| Préfixe | Phase |
|---|---|
| `01` | onboard |
| `02` | context |
| `03` | plan |
| `04` | code |
| `05` | review |
| `06` | tests |
| `07` | documentation |
| `08` | deploy |
| `09` | refactor |
| `10` | maintenance |

Cf. `.claude/rules/01-standards/1-command-structure.md` pour les conventions de frontmatter et de structure des commandes.

## MCP

Configuration MCP : `.mcp.json` à la racine, serveurs déclarés au niveau racine (pas de wrapper).

## Pas dans le scope de ce document

- Documentation des règles / agents / commandes individuels — chaque artefact porte sa propre doc dans son frontmatter ou son contenu.
- Workflows projet (collecte de corpus, table mining, etc.) — voir `architecture-tables-ml.md` et docs adjacentes.
- Installation et bootstrapping — voir `INSTALL.md`.
