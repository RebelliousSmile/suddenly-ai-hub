#!/usr/bin/env python3
"""
Ren'Py Dialogue File Selector — Sélectionne les fichiers contenant des dialogues

Scanne les fichiers .rpy ET .py d'un repo Ren'Py pour identifier ceux qui
contiennent des dialogues (character "texte") et des screens de dialogue.

Fichiers ciblés :
    - .rpy: dialogue character "texte", narration, screens de dialogue
    - .py: screens.py (Ren'Py screen language en Python), script.py avec dialogue
    
Usage:
    python scripts/crawl_rpv/select_dialogue_files.py --repo-name user/repo
    python scripts/crawl_rpv/select_dialogue_files.py --local-path /path/to/repo
    python scripts/crawl_rpv/select_dialogue_files.py --repos data/renpy-repos.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import requests

# === Patterns de détection ===

# Dialogue Ren'Py dans .rpy
RENPY_DIALOGUE_PATTERNS = {
    "character_dialogue": re.compile(r'^[a-zA-Z_]\w*\s+"[^"]{5,}"', re.MULTILINE),
    "narration": re.compile(r'^n\s+"[^"]{5,}"', re.MULTILINE),
    "say_screen": re.compile(r'screen\s+\w*say', re.MULTILINE),
    "choice_screen": re.compile(r'screen\s+\w*choice', re.MULTILINE),
}

# Patterns pour identifier les .py contenant des dialogues Ren'Py
PY_DIALOGUE_PATTERNS = {
    # screens.py de Ren'Py : contient des screens avec dialogue
    'screen_say_py': re.compile(r'screen\s+\w*say\s*\(', re.MULTILINE),
    'screen_dialog_py': re.compile(r'screen\s+\w*dialogue\s*\(', re.MULTILINE),
    'screen_choice_py': re.compile(r'screen\s+\w*choice\s*\(', re.MULTILINE),
    'vscreen_dialog_py': re.compile(r'versioned\s+screen\s+\w*dialogue', re.MULTILINE),
    
    # Python string literals qui contiennent du dialogue
    'long_string': re.compile(r'"[^"]{50,}"'),
    
    # gettext translations (fréquent dans VN)
    'gettext': re.compile(r'_\s*\(\s*["\'][^"\']{10,}["\']'),
    
    # Dialogue list/dict patterns
    'dialogue_dict': re.compile(r'\{[^}]*"[^"]{10,}"\s*:\s*"[^"]{10,}"[^}]*\}', re.DOTALL),
    
    # Character dialogue list
    'char_dialogue_list': re.compile(r'\[[\s\S]*"[^"]{10,}"[\s\S]*\]'),
    
    # Variable dialogue
    'var_dialogue': re.compile(r'(dialogue|dialog|speech|narration|line)\s*=\s*["\']', re.IGNORECASE),
    
    # Image reference for dialogue background
    'bg_reference': re.compile(r'("background"|background\s*=)\s*["\'][^"\']+\w+\w+"', re.IGNORECASE),
    
    # Character definition
    'char_def': re.compile(r'define\s+\w+\s*=\s*[A-Za-z]', re.MULTILINE),
    
    # Dialogue config
    'dialogue_config': re.compile(r'(window\s+show|say_|character\s+|default\s+\w+\s*=\s*Character)', re.MULTILINE),
}

# Patterns pour identifier les OUTILS (à exclure)
TOOL_PATTERNS = {
    'galtransl': re.compile(r'GalTransl|galtransl', re.IGNORECASE),
    'vnr': re.compile(r'VNR|visual novel reader|renpy translator', re.IGNORECASE),
    'ai4visualnovel': re.compile(r'AI4VisualNovel|ai4visualnovel', re.IGNORECASE),
    'analyzer': re.compile(r'Analyzer|analyzer|analysis', re.IGNORECASE),
    'decompiler': re.compile(r'Decompiler|decompiler|disassemble', re.IGNORECASE),
    'unpacker': re.compile(r'Unpack|unpack|extract', re.IGNORECASE),
    'patcher': re.compile(r'Patcher|patcher', re.IGNORECASE),
    'helper': re.compile(r'[Hh]elper|[Uu]tilit[y|ies]|[Tt]ool', re.IGNORECASE),
}

# Noms de fichiers typiques contenant des dialogues
DIALOGUE_FILE_PATTERNS = {
    'main_script': re.compile(r'^(main|script|story|dialogue)\.rpy$', re.IGNORECASE),
    'screens': re.compile(r'^screens\.rpy$', re.IGNORECASE),
    'py_dialogue': re.compile(r'(dialogue|say|screen|character)\.py$', re.IGNORECASE),
    'py_main': re.compile(r'^(script|main)\.py$', re.IGNORECASE),
}


class RenPyFileSelector:
    """Sélectionne les fichiers de dialogue dans un projet Ren'Py."""
    
    def __init__(self, local_path=None, repo_name=None, token=None):
        self.local_path = local_path
        self.repo_name = repo_name
        self.token = token
        self.files = []
        self.dialogue_scores = {}
        
    def scan(self):
        """Scan le repo/local et retourne les fichiers avec scores de dialogue."""
        if self.local_path:
            self._scan_local()
        elif self.repo_name:
            self._scan_github()
        else:
            raise ValueError("local_path ou repo_name requis")
        
        return sorted(
            self.dialogue_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def _scan_local(self):
        """Scan un projet local."""
        path = Path(self.local_path)
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(('.rpy', '.py')):
                    filepath = os.path.join(root, filename)
                    score = self._score_dialogue_file(filepath)
                    if score > 0:
                        self.dialogue_scores[filepath] = score
                        self.files.append({
                            'path': filepath,
                            'score': score,
                            'type': 'local'
                        })
    
    def _scan_github(self):
        """Scan un repo GitHub pour les fichiers .rpy et .py."""
        files_response = requests.get(
            f"https://api.github.com/repos/{self.repo_name}/git/trees/main?recursive=1",
            headers=self._headers()
        )
        if files_response.status_code != 200:
            # Essayer master si main échoue
            files_response = requests.get(
                f"https://api.github.com/repos/{self.repo_name}/git/trees/master?recursive=1",
                headers=self._headers()
            )
        
        if files_response.status_code == 200:
            data = files_response.json()
            tree = data.get('tree', [])
            
            for entry in tree:
                path = entry.get('path', '')
                if not path.endswith(('.rpy', '.py')):
                    continue
                
                # Vérifier si c'est un fichier dialogue
                score = self._score_filename(path)
                if score > 0:
                    self.dialogue_scores[path] = score
                    self.files.append({
                        'path': path,
                        'score': score,
                        'type': 'github',
                        'download_url': entry.get('url', ''),
                        'sha': entry.get('sha', '')
                    })
        
        # Trier par score décroissant
        self.files = sorted(self.files, key=lambda x: x['score'], reverse=True)
    
    def _score_filename(self, path):
        """Score un fichier par son nom."""
        filename = os.path.basename(path).lower()
        
        # Pattern de nom (0-100)
        for pattern_name, pattern in DIALOGUE_FILE_PATTERNS.items():
            if pattern.match(filename):
                return 50  # Base score pour le nom
        
        # Extension .rpy (60) vs .py (40)
        if path.endswith('.rpy'):
            return 60
        elif path.endswith('.py'):
            return 40
        
        return 0
    
    def _score_dialogue_file(self, filepath):
        """Score un fichier par son contenu dialogue."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if len(content) > 500000:  # Limiter aux fichiers < 500Ko
                return 0
            
            score = 0
            
            # Extensions
            if filepath.endswith('.rpy'):
                # Pattern de dialogue Ren'Py
                if RENPY_DIALOGUE_PATTERNS["character_dialogue"].search(content):
                    score += 50
                if RENPY_DIALOGUE_PATTERNS["narration"].search(content):
                    score += 30
                if RENPY_DIALOGUE_PATTERNS["say_screen"].search(content):
                    score += 20
                
            elif filepath.endswith('.py'):
                # Pattern de dialogue Python (screens.py, script.py, etc.)
                for pattern_name, pattern in PY_DIALOGUE_PATTERNS.items():
                    matches = pattern.findall(content)
                    if matches:
                        # Screens.py avec screens dialogue = haut score
                        if pattern_name in ['screen_say_py', 'screen_dialog_py', 'screen_choice_py']:
                            score += 60
                        elif pattern_name == 'gettext':
                            score += 40
                        elif pattern_name == 'text_reference':
                            score += 30
                        elif pattern_name == 'dialogue_var':
                            score += 25
                        else:
                            score += 15
            
            return score
            
        except Exception:
            return 0
    
    def get_content(self, file_info, max_size=100000):
        """Récupère le contenu d'un fichier."""
        file_type = file_info.get('type', 'github')
        path = file_info['path']
        
        if file_type == 'local':
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return content[:max_size]
            except Exception:
                return None
        
        elif file_type == 'github':
            try:
                # Utiliser le raw GitHub URL
                user_repo = self.repo_name
                raw_url = f"https://raw.githubusercontent.com/{user_repo}/main/{path}"
                
                response = requests.get(
                    raw_url,
                    headers=self._headers(),
                    timeout=10
                )
                
                if response.status_code == 200:
                    return response.text[:max_size]
                else:
                    # Essayer master
                    raw_url = f"https://raw.githubusercontent.com/{user_repo}/master/{path}"
                    response = requests.get(
                        raw_url,
                        headers=self._headers(),
                        timeout=10
                    )
                    if response.status_code == 200:
                        return response.text[:max_size]
            
            except Exception:
                return None
        
        return None
    
    def _headers(self):
        """Headers pour les requêtes GitHub."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Suddenly-RP-Model-Builder/1.0"
        }
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers


def is_tool_repo(repo_info, content=None):
    """Vérifie si un repo est un outil (pas un VN)."""
    name = repo_info.get('name', '')
    full_name = repo_info.get('full_name', '')
    description = repo_info.get('description', '')
    
    # Vérifier les patterns d'outils dans le nom ou la description
    for pattern_name, pattern in TOOL_PATTERNS.items():
        if pattern.search(name) or pattern.search(full_name) or pattern.search(description):
            return True
    
    # Vérifier dans le contenu si fourni
    if content:
        for pattern_name, pattern in TOOL_PATTERNS.items():
            if pattern.search(content):
                return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Sélectionner les fichiers dialogue d'un projet Ren'Py"
    )
    parser.add_argument("--repo-name", help="Nom du repo GitHub (user/repo)")
    parser.add_argument("--local-path", help="Chemin local du projet")
    parser.add_argument("--repos", help="Fichier JSON de repos à scanner")
    parser.add_argument("--token", help="Token GitHub (optionnel)")
    parser.add_argument("--output", help="Fichier JSON de sortie")
    parser.add_argument("--min-score", type=int, default=50, help="Score min pour inclusion")
    
    args = parser.parse_args()
    
    if not args.repo_name and not args.local_path and not args.repos:
        print("⚠️  Aucun repo spécifié. Utilisez --repo-name, --local-path ou --repos")
        return
    
    # Si on a un fichier de repos, les scanner tous
    if args.repos:
        with open(args.repos, 'r', encoding='utf-8') as f:
            repos = json.load(f)
        
        all_files = []
        for repo in repos:
            print(f"\n🔍 Scan de {repo.get('name', 'unknown')}...")
            selector = RenPyFileSelector(
                repo_name=repo.get('name', ''),
                token=args.token
            )
            
            scored_files = selector.scan()
            relevant_files = [
                (path, score) for path, score in scored_files 
                if score >= args.min_score
            ]
            
            # Vérifier si c'est un outil
            if is_tool_repo(repo):
                print(f"   ⚠️  Outil détecté, exclu")
                continue
            
            print(f"   ✅ {len(relevant_files)} fichiers dialogue trouvés")
            for path, score in relevant_files[:10]:
                print(f"      {score:3d}/100  {path}")
            
            all_files.extend([
                {
                    'repo': repo.get('name', ''),
                    'path': path,
                    'score': score,
                    'type': next(
                        (f['type'] for f in selector.files if f['path'] == path),
                        'unknown'
                    )
                }
                for path, score in relevant_files
            ])
        
        print(f"\n📊 Total: {len(all_files)} fichiers dialogue trouvés")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_files, f, indent=2, ensure_ascii=False)
            print(f"💾 Résultat sauvegardé dans {args.output}")
        
        return all_files
    
    # Scan d'un seul repo ou path
    selector = RenPyFileSelector(
        local_path=args.local_path,
        repo_name=args.repo_name,
        token=args.token
    )
    
    print(f"🔍 Scan des fichiers dialogue{' local' if args.local_path else f' dans {args.repo_name}'}...")
    
    scored_files = selector.scan()
    
    # Filtrer par score minimum
    relevant_files = [(path, score) for path, score in scored_files if score >= args.min_score]
    
    print(f"\n✅ {len(relevant_files)} fichiers trouvés (score ≥ {args.min_score})")
    
    for path, score in relevant_files[:20]:
        print(f"   {score:3d}/100  {path}")
    
    if relevant_files:
        if args.output:
            output_data = [
                {
                    'path': path,
                    'score': score,
                    'type': next(
                        (f['type'] for f in selector.files if f['path'] == path),
                        'unknown'
                    )
                }
                for path, score in relevant_files
            ]
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Résultat sauvegardé dans {args.output}")
    
    return relevant_files


if __name__ == "__main__":
    main()
