#!/usr/bin/env python3
"""
Ren'Py Corpus Pipeline — Pipeline complet de découverte, extraction et conversion

Usage:
    python scripts/crawl_rpv/run_pipeline.py --output data/renpy-corpus.jsonl
    python scripts/crawl_rpv/run_pipeline.py --max-repos 20 --genre "Romance" --situation "Relation"
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path

# Ajouter le répertoire scripts au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from github_search import discover_repos
from extract_dialogues import RenPyCorpusBuilder


def run_pipeline(output_path: str, max_repos: int = 50, genre: str = "inconnu", 
                 situation: str = "inconnu", min_tokens: int = 100, dry_run: bool = False):
    """Exécute le pipeline complet Ren'Py."""
    
    print("🚀 Pipeline Ren'Py — Corpus pour Suddenly RP")
    print("="*60)
    
    # Étape 1: Découverte des repos
    print("\n📡 Étape 1/3 : Découverte des projets Ren'Py")
    print("-"*40)
    
    if dry_run:
        print("   ⚠️  Mode dry-run: recherche limitée")
        repos = [{"name": "test-repo", "rpy_files": [{"path": "script.rpy", "url": "test"}]}]
    else:
        repos = discover_repos(max_results=max_repos)
    
    if not repos:
        print("   ❌ Aucun projet Ren'Py trouvé!")
        return
    
    print(f"   ✅ {len(repos)} projets trouvés")
    
    # Sauvegarder la liste des repos pour référence
    repos_path = "data/renpy-repos.json"
    os.makedirs("data", exist_ok=True)
    with open(repos_path, "w", encoding="utf-8") as f:
        json.dump(repos, f, indent=2, ensure_ascii=False)
    print(f"   💾 Liste des repos sauvegardée: {repos_path}")
    
    # Étape 2: Extraction des dialogues
    print("\n🔍 Étape 2/3 : Extraction des dialogues")
    print("-"*40)
    
    builder = RenPyCorpusBuilder(genre, situation)
    
    for repo in repos:
        builder.process_repo(repo)
    
    # Étape 3: Sauvegarde du corpus
    print("\n💾 Étape 3/3 : Sauvegarde du corpus")
    print("-"*40)
    
    builder.save_corpus(output_path, min_tokens)
    
    # Résumé final
    print("\n" + "="*60)
    print("🎉 Pipeline terminé!")
    print(f"   Conversations générées: {len(builder.corpus)}")
    print(f"   Fichiers traités: {builder.stats['files_processed']}")
    print(f"   Tokens générés: {builder.stats['tokens_generated']:,}")
    print(f"   Sortie: {output_path}")
    
    return builder.corpus


def main():
    parser = argparse.ArgumentParser(description="Ren'Py Corpus Pipeline")
    parser.add_argument("--output", default="data/renpy-corpus.jsonl", help="Fichier de sortie")
    parser.add_argument("--max-repos", type=int, default=50, help="Max projets à découvrir")
    parser.add_argument("--genre", default="inconnu", help="Genre Ren'Py")
    parser.add_argument("--situation", default="inconnu", help="Situation Ren'Py")
    parser.add_argument("--min-tokens", type=int, default=100, help="Min tokens par conversation")
    parser.add_argument("--dry-run", action="store_true", help="Mode test sans requêtes")
    
    args = parser.parse_args()
    
    run_pipeline(
        output_path=args.output,
        max_repos=args.max_repos,
        genre=args.genre,
        situation=args.situation,
        min_tokens=args.min_tokens,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
