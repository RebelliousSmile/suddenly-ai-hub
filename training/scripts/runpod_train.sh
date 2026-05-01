#!/usr/bin/env bash
# runpod_train.sh — Lancement d'un entraînement Axolotl sur RunPod
#
# Usage :
#   ./scripts/runpod_train.sh <config.yml>
#   ./scripts/runpod_train.sh training/suddenly-7b.yml
#   ./scripts/runpod_train.sh training/lora-univers.yml   # après substitution des placeholders
#
# Prérequis RunPod :
#   - Template : axolotl-runpod (image officielle Axolotl, Axolotl + vLLM pré-installés)
#   - GPU       : A100-40G (recommandé pour QLoRA 7B/12B)
#   - Volume    : monter un Network Volume sur /workspace pour persistance entre runs
#   - Env vars  : HF_TOKEN et éventuellement WANDB_API_KEY dans les secrets du pod
#
# Étape manuelle avant lancement :
#   1. Copier ce repo sur le pod : git clone ou upload via RunPod File Browser
#   2. Copier les JSONL du corpus dans training/data/ (depuis S3 ou upload direct)

set -euo pipefail

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
    echo "Usage: $0 <config.yml>"
    echo "  ex:  $0 training/suddenly-7b.yml"
    exit 1
fi

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/suddenly-ai-hub"

# ---------------------------------------------------------------------------
# 1. Authentification Hugging Face
#    HF_TOKEN doit être défini dans les secrets RunPod ou exporté avant ce script
# ---------------------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ERROR] HF_TOKEN non défini. Exporter la variable ou la définir dans les secrets RunPod."
    exit 1
fi
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# ---------------------------------------------------------------------------
# 2. Installation d'Axolotl (si pas dans l'image)
#    L'image axolotl-runpod l'inclut déjà. Ce bloc est un fallback.
# ---------------------------------------------------------------------------
if ! command -v axolotl &>/dev/null; then
    echo "[INFO] Axolotl non trouvé — installation depuis pip..."
    pip install axolotl[flash-attn,deepspeed] --quiet
fi

# ---------------------------------------------------------------------------
# 3. Vérifier que le config existe
# ---------------------------------------------------------------------------
CONFIG_PATH="${REPO_DIR}/${CONFIG}"
if [[ ! -f "$CONFIG_PATH" ]]; then
    # Essayer chemin absolu tel quel
    CONFIG_PATH="${CONFIG}"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Config non trouvée : ${CONFIG}"
    exit 1
fi

echo "[INFO] Config : ${CONFIG_PATH}"
echo "[INFO] GPU disponible :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi absent)"

# ---------------------------------------------------------------------------
# 4. Lancement de l'entraînement
# ---------------------------------------------------------------------------
echo "[INFO] Démarrage entraînement..."
cd "${REPO_DIR}"

axolotl train "${CONFIG_PATH}"

echo "[INFO] Entraînement terminé."

# ---------------------------------------------------------------------------
# 5. Sauvegarde vers S3 (optionnel)
#    Nécessite AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET définis
# ---------------------------------------------------------------------------
# Extraire output_dir depuis le YAML (nécessite python ou yq)
OUTPUT_DIR=$(python3 -c "
import yaml, sys
with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('output_dir', './outputs/model'))
")

if [[ -n "${S3_BUCKET:-}" ]]; then
    echo "[INFO] Upload vers S3 : s3://${S3_BUCKET}/models/$(basename ${OUTPUT_DIR})/"
    aws s3 sync "${OUTPUT_DIR}" "s3://${S3_BUCKET}/models/$(basename ${OUTPUT_DIR})/" \
        --exclude "*.tmp" \
        --no-progress
    echo "[INFO] Upload S3 terminé."
else
    echo "[INFO] S3_BUCKET non défini — sauvegarde locale uniquement dans ${OUTPUT_DIR}"
fi

# ---------------------------------------------------------------------------
# 6. Upload Hugging Face Hub (optionnel)
#    Décommenter et définir HF_REPO_ID si push vers HF Hub souhaité
# ---------------------------------------------------------------------------
# HF_REPO_ID="suddenly-ai/$(basename ${OUTPUT_DIR})"
# echo "[INFO] Push vers HF Hub : ${HF_REPO_ID}"
# huggingface-cli upload "${HF_REPO_ID}" "${OUTPUT_DIR}"

echo "[INFO] Script terminé. Penser à terminer le pod RunPod pour éviter les frais."
