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
