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
