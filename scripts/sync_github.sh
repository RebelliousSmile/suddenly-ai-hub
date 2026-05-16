#!/bin/bash
# Script pour syncroniser le code vers GitHub
# À exécuter une fois pour configurer l'authentification

echo "============================================================"
echo "🔧 Configuration GitHub pour suddenly-ai-hub"
echo "============================================================"

# Vérifier si un token GitHub est fourni
if [ -n "$GITHUB_TOKEN" ]; then
    echo "✅ Token GitHub trouvé dans les variables d'environnement"
    
    # Configurer le remote avec le token
    git remote set-url origin "https://${GITHUB_TOKEN}@github.com/RebelliousSmile/suddenly-ai-hub.git"
    echo "✅ Remote configuré avec le token"
else
    echo "❌ Aucun token GitHub trouvé"
    echo "   Tu dois définir GITHUB_TOKEN dans Railway:"
    echo "   1. Va dans Railway Dashboard"
    echo "   2. Settings > Environment Variables"
    echo "   3. Ajoute GITHUB_TOKEN=ton_token_ghp_xxxxx"
    echo "   4. Redémarre le service"
    exit 1
fi

# Push vers GitHub
echo ""
echo "🚀 Push vers GitHub..."
git push origin master

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Code syncronisé avec succès!"
    echo "📊 Voir ton repo: https://github.com/RebelliousSmile/suddenly-ai-hub"
else
    echo ""
    echo "❌ Erreur lors du push"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ Configuration terminée!"
echo "============================================================"
