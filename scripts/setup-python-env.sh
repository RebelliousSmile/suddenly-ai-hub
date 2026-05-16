#!/usr/bin/env bash
# setup-python-env.sh
# Script de mise en place de l'environnement Python pour suddenly-ai-hub
# Correspond à l'issue #44 : Étape 1 : Préparer l'environnement Python
#
# Usage: bash setup-python-env.sh [--verbose] [--skip-venv] [--skip-git]
#
# Ce script :
#   - Vérifie Python 3.10+
#   - Crée un venv dans ~/projects/suddenly-ai-hub/
#   - Installe les dépendances du projet
#   - Clone le repo si nécessaire
#   - Configure la clé API Together.ai

set -euo pipefail

VERBOSE=false
SKIP_VENV=false
SKIP_GIT=false

for arg in "$@"; do
    case "$arg" in
        --verbose|-v) VERBOSE=true ;;
        --skip-venv) SKIP_VENV=true ;;
        --skip-git) SKIP_GIT=true ;;
        --help|-h)
            echo "Usage: bash $0 [--verbose] [--skip-venv] [--skip-git]"
            echo ""
            echo "Options:"
            echo "  --verbose  Affiche les détails de chaque commande"
            echo "  --skip-venv  Saute la création du venv (utilise Python système)"
            echo "  --skip-git  Ne clone pas le repo"
            exit 0
            ;;
    esac
done

LOG_FILE="${LOG_FILE:-/tmp/suddenly-setup-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "  Setup Environment for suddenly-ai-hub"
echo "  Started: $(date)"
echo "  Log: $LOG_FILE"
echo "============================================================"
echo ""

# ---------------------------
# ÉTAPE 1: Vérifier Python
# ---------------------------
echo "📍 ÉTAPE 1: Vérifier Python 3.10+"
echo "------------------------------------"

PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD=$(command -v python3)
elif command -v python &>/dev/null; then
    PYTHON_CMD=$(command -v python)
else
    echo "❌ ERREUR: Python n'est pas installé."
    echo "   Installe-le avec: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1)
echo "   Python trouvé: $PYTHON_VERSION"
echo "   Chemin: $PYTHON_CMD"

# Vérifier version >= 3.10
MAJOR=$("$PYTHON_CMD" -c "import sys; print(sys.version_info.major)")
MINOR=$("$PYTHON_CMD" -c "import sys; print(sys.version_info.minor)")

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo "❌ ERREUR: Python 3.10+ requis, vous avez $MAJOR.$MINOR"
    echo "   Installez Python 3.10+ avec:"
    echo "     sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "     sudo apt install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi
echo "✅ Python version OK ($MAJOR.$MINOR >= 3.10)"
echo ""

# ---------------------------
# ÉTAPE 2: Git
# ---------------------------
echo "📍 ÉTAPE 2: Vérifier Git"
echo "------------------------------------"

if ! command -v git &>/dev/null; then
    echo "❌ ERREUR: Git n'est pas installé."
    echo "   Installe-le avec: sudo apt install git"
    exit 1
fi

GIT_VERSION=$(git --version)
echo "   Git trouvé: $GIT_VERSION"
echo "✅ Git OK"
echo ""

# ---------------------------
# ÉTAPE 3: Clone du repo
# ---------------------------
echo "📍 ÉTAPE 3: Repo suddenly-ai-hub"
echo "------------------------------------"

PROJECT_DIR="$HOME/projects/suddenly-ai-hub"

if [ "$SKIP_GIT" = true ]; then
    echo "⏭️  Skip (option --skip-git)"
elif [ -d "$PROJECT_DIR/.git" ]; then
    echo "   Repo déjà présent dans $PROJECT_DIR"
    cd "$PROJECT_DIR"
    CURRENT_BRANCH=$(git branch --show-current)
    CURRENT_HASH=$(git rev-parse --short HEAD)
    echo "   Branche courante: $CURRENT_BRANCH"
    echo "   Dernier commit: $CURRENT_HASH"
    
    echo "   Mise à jour du repo..."
    git fetch origin 2>/dev/null || true
    git pull --ff-only 2>/dev/null || echo "   ⚠️  Pull skipped (branch locale modifiée ?)"
    echo "✅ Repo sync OK"
else
    echo "   Clone du repo dans $PROJECT_DIR..."
    mkdir -p "$HOME/projects"
    git clone https://github.com/RebelliousSmile/suddenly-ai-hub.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    echo "✅ Repo cloné OK"
fi
echo ""

# ---------------------------
# ÉTAPE 4: Virtual Environment
# ---------------------------
echo "📍 ÉTAPE 4: Virtual Environment"
echo "------------------------------------"

VENV_DIR="$PROJECT_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python"

if [ "$SKIP_VENV" = true ]; then
    echo "⏭️  Skip (option --skip-venv)"
    echo "   Utilisation du Python système: $PYTHON_CMD"
elif [ -d "$VENV_DIR" ]; then
    echo "   Venv existant dans $VENV_DIR"
    VENV_PYTHON="$VENV_DIR/bin/python"
    
    # Vérifier que le venv est encore fonctionnel
    if "$VENV_PYTHON" --version &>/dev/null; then
        VENV_PYTHON_VER=$("$VENV_PYTHON" --version 2>&1)
        echo "   Venv OK: $VENV_PYTHON_VER"
        
        # Mettre à jour pip
        echo "   Mise à jour de pip..."
        "$VENV_PYTHON" -m pip install --upgrade pip 2>&1 | tail -1
        echo "✅ Venv mis à jour"
    else
        echo "   ⚠️  Venv corrompu, re-création..."
        rm -rf "$VENV_DIR"
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        "$VENV_PYTHON" -m pip install --upgrade pip 2>&1 | tail -1
        echo "✅ Venv re-créé"
    fi
else
    echo "   Création du venv dans $VENV_DIR..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    "$VENV_PYTHON" -m pip install --upgrade pip 2>&1 | tail -1
    echo "✅ Venv créé"
fi
echo ""

# ---------------------------
# ÉTAPE 5: Dépendances
# ---------------------------
echo "📍 ÉTAPE 5: Installer les dépendances du projet"
echo "------------------------------------"

cd "$PROJECT_DIR"

# Vérifier s'il y a un requirements.txt
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
SETUP_FILE="$PROJECT_DIR/setup.py"
PYPROJECT_FILE="$PROJECT_DIR/pyproject.toml"

DEPS_INSTALLED=false

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "   Found requirements.txt"
    "$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE" 2>&1 | tail -3 || {
        echo "❌ Échec de l'installation requirements.txt"
    }
    DEPS_INSTALLED=true
elif [ -f "$PYPROJECT_FILE" ]; then
    echo "   Found pyproject.toml, installation..."
    "$VENV_PYTHON" -m pip install ".[dev]" 2>&1 | tail -3 || {
        echo "   ⚠️  Installation complète échouée, essaye sans [dev]..."
        "$VENV_PYTHON" -m pip install . 2>&1 | tail -3 || true
    }
    DEPS_INSTALLED=true
elif [ -f "$SETUP_FILE" ]; then
    echo "   Found setup.py, installation..."
    "$VENV_PYTHON" -m pip install . 2>&1 | tail -3 || true
    DEPS_INSTALLED=true
else
    echo "   Aucun requirements.txt/pyproject.toml/setup.py trouvé."
    echo "   Installation des dépendances minimales du projet..."
    echo ""
    echo "   [1/6] requests + beautifulsoup4 (pour le scraping #47)..."
    "$VENV_PYTHON" -m pip install requests beautifulsoup4 2>&1 | tail -1
    echo "   [2/6] datasets (HuggingFace, pour la gestion des datasets)..."
    "$VENV_PYTHON" -m pip install datasets 2>&1 | tail -1
    echo "   [3/6] axolotl (pour le fine-tuning #50)..."
    echo "   ⏭️  Skipped (axolotl a beaucoup de dépendances, installe-le séparément si besoin)"
    echo "       pip install axolotl-ai 2>&1 | tail -1"
    echo "   [4/6] together (pour l'API Together.ai #45)..."
    "$VENV_PYTHON" -m pip install together 2>&1 | tail -1
    echo "   [5/6] python-dotenv (pour les variables d'environnement)..."
    "$VENV_PYTHON" -m pip install python-dotenv 2>&1 | tail -1
    echo "   [6/6] pytest (pour les tests)..."
    "$VENV_PYTHON" -m pip install pytest 2>&1 | tail -1
    echo ""
    echo "✅ Dépendances minimales installées"
fi
echo ""

# ---------------------------
# ÉTAPE 6: Together.ai API Key
# ---------------------------
echo "📍 ÉTAPE 6: Configurer Together.ai API Key"
echo "------------------------------------"

if [ -f "$PROJECT_DIR/.env" ] && grep -q "TOGETHER_API_KEY" "$PROJECT_DIR/.env"; then
    echo "   ✅ TOGETHER_API_KEY déjà configuré dans .env"
    # Vérifier que la clé n'est pas vide
    API_KEY=$(grep TOGETHER_API_KEY "$PROJECT_DIR/.env" | cut -d= -f2 | tr -d '"')
    if [ -n "$API_KEY" ] && [ "$API_KEY" != "your-api-key-here" ]; then
        MASKED="${API_KEY:0:5}...${API_KEY: -4}"
        echo "   Clé configurée: $MASKED"
    else
        echo "   ⚠️  La clé est vide ou au format par défaut."
        echo "   Édite .env et remplace TOGETHER_API_KEY=par_ta_clé"
    fi
else
    echo "   ❌ TOGETHER_API_KEY non configuré"
    echo ""
    echo "   Crée un fichier .env à la racine du projet:"
    echo ""
    echo "   cd $PROJECT_DIR"
    echo "   echo 'TOGETHER_API_KEY=ta_clé_ici' > .env"
    echo ""
    echo "   Obtiens ta clé sur https://api.together.xyz/settings/api-keys"
    echo "   (Crédits: $30 de gratuit)"
    echo ""
    
    # Créer un .env.template pour aider
    if [ ! -f "$PROJECT_DIR/.env.example" ] && [ ! -f "$PROJECT_DIR/.env.template" ]; then
        cat > "$PROJECT_DIR/.env.example" << 'EOF'
# Together.ai API Key (voir https://api.together.xyz/settings/api-keys)
TOGETHER_API_KEY=ta_v1_your_api_key_here

# RunPod API Key (pour le déploiement #17)
RUNPOD_API_KEY=

# Hetzner/VPS credentials (gateway #17)
HETZNER_API_TOKEN=

# PostgreSQL credentials (corpus #16)
DATABASE_URL=postgresql://user:pass@localhost:5432/suddenly

# S3-compatible (corpus #16)
S3_ENDPOINT=
S3_BUCKET=
S3_ACCESS_KEY=
S3_SECRET_KEY=
EOF
        echo "✅ .env.example créé (copie-le en .env et remplis tes valeurs)"
    fi
fi
echo ""

# ---------------------------
# VÉRIFICATION FINALE
# ---------------------------
echo "📍 ÉTAPE 7: Vérification finale"
echo "------------------------------------"

echo ""
echo "Résumé de l'environnement:"
echo "  Python : $( "$VENV_PYTHON" --version 2>&1 )"
echo "  Pip    : $("$VENV_PYTHON" -m pip --version 2>&1)"
echo "  Venv   : $VENV_DIR"
echo "  Repo   : $PROJECT_DIR"
echo "  Branch : $(git branch --show-current 2>/dev/null || echo '(non git)')"
echo ""

# Lister les paquets installés
echo "Paquets principaux installés:"
"$VENV_PYTHON" -m pip list 2>/dev/null | grep -iE "requests|beautifulsoup|datasets|together|dotenv|pytest|axolotl" || echo "  (aucun des paquets ciblés trouvé)"
echo ""

# Test rapide d'import
echo "Test d'import Python:"
$VENV_PYTHON -c "import sys; print(f'  Python {sys.version} OK')"
$VENV_PYTHON -c "import requests; print(f'  requests {requests.__version__} OK')"
$VENV_PYTHON -c "import bs4; print(f'  beautifulsoup4 {bs4.__version__} OK')"
$VENV_PYTHON -c "import datasets; print(f'  datasets {datasets.__version__} OK')"
$VENV_PYTHON -c "import together; print(f'  together OK')" || echo "  ⚠️  together non installé"
$VENV_PYTHON -c "import dotenv; print(f'  python-dotenv {dotenv.__version__} OK')" || echo "  ⚠️  python-dotenv non installé"
$VENV_PYTHON -c "import pytest; print(f'  pytest {pytest.__version__} OK')" || echo "  ⚠️  pytest non installé"
echo ""

echo "============================================================"
echo "  ✅ Setup terminé ! $(date)"
echo "============================================================"
echo ""
echo "Prochaines étapes recommandées:"
echo "  1. Remplir .env avec ta clé Together.ai"
echo "  2. Tester l'API: bash tests/test_api_together.sh (à créer)"
echo "  3. Passer à #46 : créer le dataset test (10 exemples)"
echo ""
echo "Pour activer le venv dans un shell:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Log complet: $LOG_FILE"
