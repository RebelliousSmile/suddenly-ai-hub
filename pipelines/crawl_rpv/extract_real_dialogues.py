#!/usr/bin/env python3
"""
Extracteur de dialogues depuis fichiers .rpy Ren'Py (local)

Lit les fichiers .rpy d'un repo cloné et extrait les dialogues
au format JSONL Axolotl pour fine-tuning.
"""

import re
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class RPyDialogueExtractor:
    """Extrait les dialogues depuis les fichiers .rpy."""
    
    # Regex pour les dialogues character "texte"
    CHAR_DIALOGUE = re.compile(
        r'^\s*([a-zA-Z_]\w*)\s+"([^"]*)"', re.MULTILINE
    )
    
    # Regex pour les définitions de caractères
    DEFINE_CHAR = re.compile(
        r'^\s*define\s+(\w+)\s*=\s*Character\((.+?)\)', re.MULTILINE
    )
    
    # Regex pour les labels
    LABEL = re.compile(r'^\s*label\s+(\w+)', re.MULTILINE)
    
    # Mots-clés Ren'Py (non-dialogue)
    KEYWORDS = {
        'menu', 'if', 'elif', 'else', 'jump', 'call', 'return',
        'show', 'hide', 'pause', 'play', 'stop', 'label', 'default',
        'define', 'init', 'python', 'image', 'transform', 'key',
        'screen', 'use', 'clone', 'add', 'at', 'layer', 'zorder',
        'with', 'prefer', 'block', 'replace', 'key', 'action',
        'textbutton', 'vbox', 'hbox', 'frame', 'viewport', 'side',
        'bar', 'radiobutton', 'checkbutton', 'textbox', 'window',
        'has', 'prefs', 'qt', 'say', 'null', 'true', 'false',
        'len', 'min', 'max', 'int', 'float', 'str', 'abs', 'round',
        'range', 'enumerate', 'reversed', 'sorted', 'zip', 'map',
        'filter', 'any', 'all', 'set', 'list', 'tuple', 'dict',
    }
    
    def __init__(self):
        self.characters = {}  # nom -> display_name
    
    def parse_characters(self, content: str):
        """Parse les définitions de caractères (define e = Character("Eileen"))."""
        for match in self.DEFINE_CHAR.finditer(content):
            char_id = match.group(1)
            char_args = match.group(2)
            
            # Extraire le nom affiché entre guillemets
            name_match = re.search(r'["\'](.+?)["\']', char_args)
            display_name = name_match.group(1) if name_match else char_id
            
            self.characters[char_id] = display_name
    
    def extract_dialogues(self, content: str) -> List[Tuple[str, str]]:
        """Extrait les dialogues character "texte"."""
        dialogues = []
        
        for match in self.CHAR_DIALOGUE.finditer(content):
            char_id = match.group(1)
            text = match.group(2)
            
            # Ignorer les mots-clés Ren'Py
            if char_id.lower() in self.KEYWORDS:
                continue
            
            # Remplacer le char_id par le display_name si connu
            display_name = self.characters.get(char_id, char_id)
            
            if text.strip():  # Ne pas garder les lignes vides
                dialogues.append((display_name, text))
        
        return dialogues
    
    def extract_screen_choices(self, content: str) -> List[str]:
        """Extrait les choix menu dans les screens."""
        choices = []
        
        # Pattern: "texte" action Jump("label")
        choice_pattern = re.compile(
            r'^\s*textbutton\s+"([^"]+)"\s+action\s+(\w+)', re.MULTILINE
        )
        
        for match in choice_pattern.finditer(content):
            text = match.group(1)
            action = match.group(2)
            if text.strip() and action in ('Jump', 'Call', 'Return'):
                choices.append(f"Choice: {text}")
        
        return choices


def extract_from_rpy_files(rpy_dir: str) -> Dict:
    """Extrait les dialogues depuis tous les fichiers .rpy d'un répertoire."""
    extractor = RPyDialogueExtractor()
    
    rpy_path = Path(rpy_dir)
    all_dialogues = []
    all_characters = {}
    labels = []
    
    # Trouver tous les fichiers .rpy
    rpy_files = list(rpy_path.rglob('*.rpy')) + list(rpy_path.rglob('*.rpyc'))
    rpy_files = [f for f in rpy_files if f.is_file()]
    
    if not rpy_files:
        print(f"  ⚠️  Aucun fichier .rpy trouvé dans {rpy_dir}")
        return None
    
    print(f"  📁 {len(rpy_files)} fichiers .rpy trouvés")
    
    # Phase 1: Parse les définitions de caractères
    for rpy_file in rpy_files:
        try:
            with open(rpy_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            extractor.parse_characters(content)
            
            # Extraire les labels
            for match in RPyDialogueExtractor.LABEL.finditer(content):
                labels.append(f"{rpy_file.name}:{match.group(1)}")
        
        except (UnicodeDecodeError, PermissionError) as e:
            print(f"    ⚠️  {rpy_file.name}: {e}")
            continue
    
    print(f"  👥 {len(extractor.characters)} personnages définis: {', '.join(extractor.characters.values())[:80]}")
    print(f"  🏷️  {len(labels)} labels trouvés")
    
    # Phase 2: Extrait les dialogues
    dialogue_count = 0
    for rpy_file in rpy_files:
        try:
            with open(rpy_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Ignorer les gros fichiers de screens (trop de UI, pas de dialogue)
            if rpy_file.name in ('screens.rpy', 'gui.rpy', 'options.rpy', 'about.rpy'):
                continue
            
            dialogues = extractor.extract_dialogues(content)
            if dialogues:
                all_dialogues.extend(dialogues)
                dialogue_count += len(dialogues)
        
        except (UnicodeDecodeError, PermissionError):
            continue
    
    print(f"  💬 {dialogue_count} dialogues extraits")
    
    if not all_dialogues:
        return None
    
    # Créer un personnage "narrateur" par défaut
    if not all_characters:
        all_characters["narrator"] = "Narrateur"
    
    return {
        "dialogues": all_dialogues,
        "characters": extractor.characters,
        "labels": labels[:20],  # Limiter pour ne pas surcharger
        "file_count": len(rpy_files),
    }


def create_axolotl_entry(dialogues: List[Tuple[str, str]], characters: Dict) -> Dict:
    """Crée un entry Axolotl à partir des dialogues extraits."""
    # Prendre les N premiers dialogues pour créer un échange cohérent
    n_dialogues = min(20, len(dialogues))
    selected = dialogues[:n_dialogues]
    
    messages = []
    
    # Message système avec contexte
    char_names = list(characters.values())[:5]
    system_content = (
        f"Scène d'un visual novel Ren'Py. "
        f"Personnages: {', '.join(char_names)}. "
        f"Dialogue en français."
    )
    messages.append({
        "role": "system",
        "content": system_content,
    })
    
    # Messages alternés user/assistant
    for i, (char, text) in enumerate(selected):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({
            "role": role,
            "content": f"{char}: {text}",
        })
    
    return {
        "messages": messages,
        "metadata": {
            "source": "renpy-real",
            "characters": char_names,
            "dialogue_count": len(selected),
            "type": "visual_novel",
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extraire des dialogues depuis des fichiers .rpy Ren'Py locaux"
    )
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Répertoire contenant les fichiers .rpy (ex: /tmp/LearnToCodeRPG/game)'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/renpy-real-dialogues.jsonl',
        help='Fichier de sortie JSONL (default: data/renpy-real-dialogues.jsonl)'
    )
    parser.add_argument(
        '--entries', '-n',
        type=int,
        default=100,
        help='Nombre max d\'entries à générer (default: 100)'
    )
    
    args = parser.parse_args()
    
    print(f"📂 Lecture de {args.input_dir}...")
    result = extract_from_rpy_files(args.input_dir)
    
    if not result:
        print("❌ Aucun dialogue trouvé.")
        return
    
    # Créer les entries Axolotl
    print(f"\n🔄 Création de {args.entries} entries Axolotl...")
    entries = []
    
    for i in range(args.entries):
        # Créer des variations en mélangeant les dialogues
        import random
        shuffled = result['dialogues'][:]
        random.shuffle(shuffled)
        
        entry = create_axolotl_entry(shuffled, result['characters'])
        entry['metadata']['entry_id'] = i
        entries.append(entry)
    
    # Sauvegarder
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')
    
    print(f"\n✅ {len(entries)} entries sauvegardées dans {args.output}")
    print(f"📊 Statistiques:")
    print(f"   Fichiers .rpy: {result['file_count']}")
    print(f"   Personnages: {len(result['characters'])}")
    print(f"   Dialogues extraits: {len(result['dialogues'])}")


if __name__ == '__main__':
    main()
