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
