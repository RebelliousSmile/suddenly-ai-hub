#!/usr/bin/env python3
"""
Ren'Py Dialogue Extractor — Extraction de dialogues depuis fichiers .rpy

Parse les fichiers Ren'Py et extrait les dialogues structurés en format JSONL
compatible Axolotl pour fine-tuning.

Usage:
    python scripts/crawl_rpv/extract_dialogues.py --repos data/renpy-repos.json --output data/renpy-corpus.jsonl
"""

import json
import re
import os
import random
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
from collections import Counter


class RPYParser:
    """Parseur de fichiers .rpy Ren'Py."""
    
    # Regex pour extraire les lignes de dialogue
    # Format: `label "..."` ou `label "..."` avec caractères spéciaux
    DIALOGUE_PATTERN = re.compile(
        r'^\s*(?:\w+)\s+"([^"]+)"',  # label "dialogue"
        re.MULTILINE
    )
    
    # Format avec préfixe de personnage: `name "dialogue"`
    CHARACTER_DIALOGUE_PATTERN = re.compile(
        r'^\s*([a-zA-Z_][\w]*)\s+"([^"]+)"',
        re.MULTILINE
    )
    
    # Labels de scène
    LABEL_PATTERN = re.compile(r'^\s*label\s+(\w+)', re.MULTILINE)
    SHOW_PATTERN = re.compile(r'^\s*show\s+(.+)', re.MULTILINE)
    
    # Mots-clés Ren'Py à ignorer (doit être complet!)
    RENPY_KEYWORDS = {
        "menu", "if", "elif", "else", "jump", "call", "return",
        "show", "hide", "pause", "play", "stop", "label", "default",
        "define", "init", "python", "image", "transform", "key",
        "screen", "use", "clone", "add", "at", "layer", "zorder",
        "with", "prefer", "timer", "timer", "timer", "timer"
    }
    SCENE_PATTERN = re.compile(r'^\s*scene\s+(\w+)', re.MULTILINE)
    NARRATION_PATTERN = re.compile(r'^\s*n\s+"([^"]+)"', re.MULTILINE)
    
    def __init__(self, strict=True):
        self.strict = strict  # Mode strict: exiger un dialogue de personnage
    
    def parse_file(self, content: str) -> Dict:
        """Parse un fichier .rpy et retourne les dialogues structurés."""
        scenes = []
        current_scene = {
            "label": None,
            "characters": [],
            "dialogues": [],
            "narrations": [],
            "context": []
        }
        
        lines = content.split("\n")
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Label de scène
            label_match = self.LABEL_PATTERN.match(line)
            if label_match:
                if current_scene["dialogues"] or current_scene["narrations"]:
                    scenes.append(self._finalize_scene(current_scene))
                    current_scene = {
                        "label": None,
                        "characters": [],
                        "dialogues": [],
                        "narrations": [],
                        "context": []
                    }
                current_scene["label"] = label_match.group(1)
            
            # Show (décor/character)
            show_match = self.SHOW_PATTERN.match(line)
            if show_match:
                current_scene["context"].append(f"show {show_match.group(1)}")
            
            # Scene (background)
            scene_match = self.SCENE_PATTERN.match(line)
            if scene_match:
                current_scene["context"].append(f"scene {scene_match.group(1)}")
            
            # Narration (n "texte") — vérifier AVANT dialogue pour éviter collision
            narr_match = self.NARRATION_PATTERN.match(line)
            if narr_match:
                current_scene["narrations"].append(narr_match.group(1))
            else:
                # Dialogue de personnage (label + "dialogue")
                char_match = self.CHARACTER_DIALOGUE_PATTERN.match(line)
                if char_match:
                    name = char_match.group(1)
                    text = char_match.group(2)
                    
                    # Ignorer les mots-clés Ren'Py
                    if name not in self.RENPY_KEYWORDS:
                        current_scene["characters"].append(name)
                        current_scene["dialogues"].append({
                            "character": name,
                            "text": text
                        })
            
            i += 1
        
        # Dernière scène
        if current_scene["dialogues"] or current_scene["narrations"]:
            scenes.append(self._finalize_scene(current_scene))
        
        return {
            "scenes": scenes,
            "total_dialogues": sum(len(s["dialogues"]) for s in scenes),
            "total_characters": set(sum([s["characters"] for s in scenes], []))
        }
    
    def _finalize_scene(self, scene: Dict) -> Dict:
        """Finalise une scène et retourne un dictionnaire propre."""
        return {
            "label": scene["label"],
            "characters": list(set(scene["characters"])),
            "dialogues": scene["dialogues"],
            "narrations": scene["narrations"],
            "context": scene["context"]
        }


class DialogueConverter:
    """Convertit les dialogues Ren'Py vers format JSONL Axolotl."""
    
    def __init__(self, genre: str = "inconnu", situation: str = "inconnu"):
        self.genre = genre
        self.situation = situation
        self.name_anonymizer = NameAnonymizer()
    
    def convert_scene(self, scene: Dict) -> Optional[Dict]:
        """Convertit une scène Ren'Py en message JSONL."""
        if not scene["dialogues"]:
            return None
        
        # Anonymiser les noms de personnages
        mapped_names = {}
        for dialogue in scene["dialogues"]:
            char_name = dialogue["character"]
            if char_name not in mapped_names:
                mapped_names[char_name] = f"Personnage{len(mapped_names)+1}"
        
        # Construire le système (contexte)
        context_parts = []
        if scene["context"]:
            context_parts.append(" | ".join(scene["context"]))
        
        context_parts.append(f"Genre: {self.genre}")
        context_parts.append(f"Situation: {self.situation}")
        
        system_message = "Contexte: " + ", ".join(context_parts)
        
        # Construire les messages conversationnels
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        for dialogue in scene["dialogues"]:
            original_name = dialogue["character"]
            anonymized_name = mapped_names.get(original_name, original_name)
            
            # Déterminer le rôle (user ou assistant)
            # Alternance entre user et assistant
            role = "user" if len(messages) % 2 == 1 else "assistant"
            
            messages.append({
                "role": role,
                "content": dialogue["text"]
            })
        
        return {
            "messages": messages,
            "metadata": {
                "source": "renpy",
                "genre": self.genre,
                "situation": self.situation,
                "label": scene["label"],
                "dialogues_count": len(scene["dialogues"]),
                "characters": list(set(mapped_names.values()))
            }
        }


class NameAnonymizer:
    """Anonymise les noms de personnages Ren'Py."""
    
    def __init__(self):
        self.mapping = {}
        self.prefix = "PNJ"  # Personnage Non Joueur
    
    def anonymize_name(self, name: str) -> str:
        """Anonymise un nom de personnage."""
        if name not in self.mapping:
            self.mapping[name] = f"{self.prefix}{len(self.mapping)+1}"
        return self.mapping[name]


class RenPyCorpusBuilder:
    """Bâtit le corpus JSONL à partir de projets Ren'Py."""
    
    def __init__(self, genre: str = "inconnu", situation: str = "inconnu"):
        self.parser = RPYParser()
        self.converter = DialogueConverter(genre, situation)
        self.corpus = []
        self.stats = {
            "files_processed": 0,
            "scenes_extracted": 0,
            "conversations_created": 0,
            "characters_anonymized": 0,
            "tokens_generated": 0
        }
    
    def process_repo(self, repo: Dict) -> List[Dict]:
        """Traite un repo Ren'Py et retourne les conversations extraites."""
        repo_name = repo.get("name", "unknown")
        rpy_files = repo.get("rpy_files", [])
        
        print(f"\n📁 Traitement du repo {repo_name} ({len(rpy_files)} fichiers .rpy)")
        
        for file_info in rpy_files[:5]:  # Limiter à 5 fichiers par repo
            file_path = file_info["path"]
            file_url = file_info["url"]
            
            try:
                content = self._download_file(file_url)
                if not content:
                    print(f"   ⏭️  {file_path}: fichier trop gros ou inaccessible")
                    continue
                
                print(f"   📄 {file_path} ({len(content)} octets)")
                
                # Parser
                parsed = self.parser.parse_file(content)
                
                if parsed["total_dialogues"] == 0:
                    print(f"   ⏭️  Pas de dialogue dans {file_path}")
                    continue
                
                print(f"   ✅ {parsed['total_dialogues']} dialogues, {len(parsed['total_characters'])} personnages")
                
                # Convertir les scènes
                for scene in parsed["scenes"]:
                    if len(scene["dialogues"]) < 2:  # Minimun 2 dialogues
                        continue
                    
                    converted = self.converter.convert_scene(scene)
                    if converted:
                        self.corpus.append(converted)
                        self.stats["conversations_created"] += 1
                        self.stats["tokens_generated"] += len(content) // 4
                
                self.stats["files_processed"] += 1
                self.stats["scenes_extracted"] += len(parsed["scenes"])
                
                time.sleep(0.5)  # Rate limiting GitHub
                
            except Exception as e:
                print(f"   ❌ Erreur {file_path}: {str(e)[:50]}")
                continue
        
        return self.corpus[-len(rpy_files):]
    
    def _download_file(self, file_url: str) -> Optional[str]:
        """Télécharge le contenu d'un fichier GitHub."""
        try:
            response = requests.get(file_url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException:
            return None
    
    def save_corpus(self, output_path: str, min_tokens: int = 100):
        """Sauvegarde le corpus en JSONL."""
        # Filtrer les conversations trop courtes
        filtered = []
        for entry in self.corpus:
            tokens = sum(len(msg["content"]) for msg in entry["messages"])
            if tokens >= min_tokens:
                filtered.append(entry)
        
        # Sauvegarde
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in filtered:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"\n💾 {len(filtered)} conversations sauvegardées dans {output_path}")
        print(f"   Tokens totaux: {self.stats['tokens_generated']:,}")
        
        # Statistiques
        print(f"\n📊 Statistiques:")
        print(f"   Fichiers traités: {self.stats['files_processed']}")
        print(f"   Scènes extraites: {self.stats['scenes_extracted']}")
        print(f"   Conversations créées: {self.stats['conversations_created']}")
        print(f"   Tokens générés: {self.stats['tokens_generated']:,}")
        print(f"   Conversations après filtrage: {len(filtered)}")


def main():
    parser = argparse.ArgumentParser(description="Extract dialogue from Ren'Py .rpy files")
    parser.add_argument("--repos", required=True, help="Fichier JSON des repos Ren'Py")
    parser.add_argument("--output", required=True, help="Fichier JSONL de sortie")
    parser.add_argument("--genre", default="inconnu", help="Genre Ren'Py")
    parser.add_argument("--situation", default="inconnu", help="Situation Ren'Py")
    parser.add_argument("--min-tokens", type=int, default=100, help="Min tokens par conversation")
    parser.add_argument("--max-repos", type=int, default=10, help="Max repos à traiter")
    
    args = parser.parse_args()
    
    # Charger les repos
    with open(args.repos, "r", encoding="utf-8") as f:
        repos = json.load(f)
    
    print(f"🔧 Chargeur de corpus Ren'Py")
    print(f"   Repos à traiter: {len(repos[:args.max_repos])}")
    print(f"   Sortie: {args.output}")
    
    # Traiter les repos
    builder = RenPyCorpusBuilder(args.genre, args.situation)
    
    for repo in repos[:args.max_repos]:
        builder.process_repo(repo)
    
    # Sauvegarder
    builder.save_corpus(args.output, args.min_tokens)
    
    print("\n✅ Corpus Ren'Py construit avec succès!")


if __name__ == "__main__":
    main()
