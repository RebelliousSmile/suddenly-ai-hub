#!/usr/bin/env python3
"""
Ren'Py Corpus Pipeline — Pipeline complet de découverte, extraction et conversion

Usage:
    python scripts/crawl_rpv/run_pipeline.py --output data/renpy-corpus.jsonl
    python scripts/crawl_rpv/run_pipeline.py --max-repos 20 --genre "Horreur" --situation "Surnaturel"
"""

import json
import os
import sys
import argparse
from pathlib import Path

# Ajouter le répertoire scripts au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from github_search import discover_repos
from extract_dialogues import RenPyCorpusBuilder


def detect_genre_from_repo(repo_name: str) -> tuple:
    """Détecte le genre et la situation à partir du nom du repo."""
    name_lower = repo_name.lower()
    
    # Cas spécifiques connus
    if "katawa" in name_lower or "shoujo" in name_lower:
        return "romance", "scolaire"
    if "ddlc" in name_lower or "dolly" in name_lower:
        return "horreur", "psychologique"
    if "learn" in name_lower and ("code" in name_lower or "rpg" in name_lower):
        return "instruction", "apprentissage"
    if "bytesoflove" in name_lower or "byte" in name_lower:
        return "romance", "scolaire"
    if "danse" in name_lower and "macabre" in name_lower:
        return "horreur", "surnaturel"
    if "visualnovel" in name_lower and "kit" in name_lower:
        return "instruction", "développement"
    if "renpy" in name_lower or "ren'py" in name_lower:
        return "instruction", "développement"
    if "dynamicsprites" in name_lower:
        return "instruction", "développement"
    if "enhanced" in name_lower and "inventory" in name_lower:
        return "instruction", "développement"
    
    # Horreur
    if any(k in name_lower for k in ["horror", "horreur", "scary", "ghost", "monster", "creep", "dark"]):
        return "horreur", "surnaturel"
    
    # Romance
    if any(k in name_lower for k in ["romance", "love", "amour", "loveletter"]):
        return "romance", "relationnelle"
    
    # Polar / Mystère
    if any(k in name_lower for k in ["mystery", "detective", "crime", "murder", "polar"]):
        return "polar", "enquête"
    
    # Science-fiction
    if any(k in name_lower for k in ["scifi", "sci-fi", "space", "alien", "futur", "cyber"]):
        return "science-fiction", "technologique"
    
    # Fantasy
    if any(k in name_lower for k in ["fantasy", "mage", "wizard", "quest", "epique"]):
        return "fantaisie", "aventure"
    
    # Steampunk
    if any(k in name_lower for k in ["steampunk", "clock", "gear"]):
        return "steampunk", "industriel"
    
    # Slice of Life
    if any(k in name_lower for k in ["school", "college", "academy", "high", "student", "life"]):
        return "slice of life", "scolaire"
    
    # Aventure
    if any(k in name_lower for k in ["adventure", "quest", "journey", "explore"]):
        return "aventure", "exploration"
    
    # Défaut
    return "inconnu", "inconnu"


def run_pipeline(output_path: str, max_repos: int = 20, min_tokens: int = 100, dry_run: bool = False):
    """Exécute le pipeline complet Ren'Py."""
    
    print("🚀 Pipeline Ren'Py — Corpus pour Suddenly RP")
    print("=" * 60)
    
    # Étape 1: Découverte des repos
    print("\n📡 Étape 1/3 : Découverte des projets Ren'Py")
    print("-" * 40)
    
    if dry_run:
        print("   ⚠️  Mode dry-run: recherche limitée")
        repos = [{"name": "test-repo", "rpy_files": [{"path": "script.rpy", "url": "test"}]}]
    else:
        repos = discover_repos(max_results=max_repos)
    
    if not repos:
        print("   ❌ Aucun projet Ren'Py trouvé!")
        return
    
    print(f"   ✅ {len(repos)} projets trouvés")
    
    # Trier par étoiles (plus populaires d'abord) et détecter genres
    repos_sorted = sorted(repos, key=lambda r: r.get("stars", 0), reverse=True)
    
    genres_found = {}
    for repo in repos_sorted:
        genre, situation = detect_genre_from_repo(repo.get("name", ""))
        repo["genre"] = genre
        repo["situation"] = situation
        genres_found[genre] = genres_found.get(genre, 0) + 1
        print(f"   📁 {repo['name']} ({repo.get('stars', 0)}⭐) → {genre} / {situation} ({len(repo.get('rpy_files', []))} fichiers)")
    
    print(f"\n   📊 Genres trouvés: {', '.join(f'{g}: {c}' for g, c in sorted(genres_found.items()))}")
    
    # Sauvegarder la liste des repos pour référence
    repos_path = "data/renpy-repos.json"
    os.makedirs("data", exist_ok=True)
    with open(repos_path, "w", encoding="utf-8") as f:
        json.dump(repos_sorted, f, indent=2, ensure_ascii=False)
    print(f"   💾 Liste des repos sauvegardée: {repos_path}")
    
    # Étape 2: Extraction des dialogues
    print("\n🔍 Étape 2/3 : Extraction des dialogues")
    print("-" * 40)
    
    builder = RenPyCorpusBuilder()
    
    for repo in repos_sorted:
        genre = repo.get("genre", "inconnu")
        situation = repo.get("situation", "inconnu")
        print(f"\n   📝 [{genre}/{situation}] Traitement de {repo['name']}...")
        builder.process_repo(repo, genre=genre, situation=situation)
    
    # Étape 3: Sauvegarde du corpus
    print("\n💾 Étape 3/3 : Sauvegarde du corpus")
    print("-" * 40)
    
    builder.save_corpus(output_path, min_tokens)
    
    # Résumé final
    print("\n" + "=" * 60)
    print("🎉 Pipeline terminé!")
    print(f"   Conversations générées: {len(builder.corpus)}")
    print(f"   Fichiers traités: {builder.stats['files_processed']}")
    print(f"   Tokens générés: {builder.stats['tokens_generated']:,}")
    print(f"   Sortie: {output_path}")
    
    return builder.corpus


def main():
    parser = argparse.ArgumentParser(description="Ren'Py Corpus Pipeline")
    parser.add_argument("--output", default="data/renpy-corpus.jsonl", help="Fichier de sortie")
    parser.add_argument("--max-repos", type=int, default=20, help="Max projets à découvrir")
    parser.add_argument("--min-tokens", type=int, default=100, help="Min tokens par conversation")
    parser.add_argument("--dry-run", action="store_true", help="Mode test sans requêtes")
    
    args = parser.parse_args()
    
    run_pipeline(
        output_path=args.output,
        max_repos=args.max_repos,
        min_tokens=args.min_tokens,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
