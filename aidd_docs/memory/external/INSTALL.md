# Installation rapide — suddenly-muses

Bootstrap de l'environnement de développement pour le projet **suddenly-muses**, branché sur aidd-framework via `aidd-custom` et `aidd-overlay`.

## Prérequis

- Python 3.11 ou supérieur (pour le service Muses et les pipelines de mining de corpus)
- Node.js 18 ou supérieur, npm (pour la CLI `aidd-custom`)
- Accès GitHub (cloner les overlays aidd)
- Git

## Étape 1 — Installer la CLI aidd-custom

```bash
cd <repo-parent>
git clone https://github.com/RebelliousSmile/aidd-custom.git
cd aidd-custom
npm install
npm link
aidd-custom --version
```

## Étape 2 — Appliquer l'overlay dans le projet

```bash
cd <repo-parent>/suddenly-muses
aidd-custom setup --repo RebelliousSmile/aidd-overlay
aidd-custom install
aidd-custom doctor
```

`aidd-custom doctor` vérifie que les règles, agents, skills et commandes sont en place dans `.claude/` selon les conventions documentées dans `AIDD_INTEGRATION.md`.

## Étape 3 — Environnement Python du projet

```bash
cd <repo-parent>/suddenly-muses
python -m venv venv
source venv/bin/activate
pip install -e .
```

Les dépendances sont déclarées dans `pyproject.toml` (extras `[pipelines, scraper, dev]`).

## Vérification

```bash
aidd-custom --version
aidd-custom doctor
python -c "import pipelines.anonymization; print('ok')"
```

## Variables d'environnement

Le service Muses n'utilise pas d'API d'inférence commerciale (cf. `philosophy.md` §7). Aucune clé `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`, `OPENAI_API_KEY`, etc. n'est requise.

Variables à définir dans `.env` (selon usage) :

```bash
GITHUB_TOKEN=...
```

D'autres variables (signature ActivityPub, accès au stockage des tables) seront définies dans `infrastructure.md` à venir.

## Dépannage

**`aidd-custom` non trouvé**

```bash
export PATH="$HOME/.npm-global/bin:$PATH"
```

ou utiliser le chemin complet binaire produit par `npm link`.

**Permissions npm**

```bash
sudo chown -R $(whoami) ~/.npm
```

## Prochaines étapes

1. Installation terminée (cette page).
2. Lire `philosophy.md` pour comprendre l'identité du projet.
3. Lire `architecture-tables-ml.md` pour le pipeline 4-étages.
4. Lire `AIDD_INTEGRATION.md` pour les conventions aidd.
