#!/usr/bin/env python3
"""
Traduction et adaptation RP de corpus Ren'Py en français — version sync robuste

Traduit les dialogues Ren'Py anglais vers un français de jeu de rôle naturel.
Utilise Llama 3.3 70B via Together AI.

Usage:
    python scripts/crawl_rpv/translate_rpv_sync.py \
        --input data/renpy-corpus-final.jsonl \
        --output data/renpy-corpus-fr.jsonl
"""

import json
import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import httpx

# Auto-load .env
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
API_KEY = os.getenv("TOGETHER_API_KEY", "")
BATCH_DELAY = 0.5  # secondes entre les requêtes
MAX_WORKERS = 8  # Requêtes parallèles maximales
MAX_RETRIES = 2
TIMEOUT = 300  # secondes par requête

# ============================================================
# PROMPT SYSTEM
# ============================================================

SYSTEM_PROMPT = """Tu es un traducteur spécialisé en jeux de rôle textuels français.
Traduire et adapter un dialogue de Visual Novel (Ren'Py) de l'anglais vers un français de jeu de rôle naturel.
Utilise le tiret français (-) et les guillemets français (« »).
Conserve la structure JSON originale. Ne fournis aucun commentaire ni markdown.
Commence toujours par {{ et termine par }}."""


def build_prompt(entry: dict) -> tuple:
    """Construit le prompt pour une entrée."""
    genre = entry.get("metadata", {}).get("genre", "inconnu")
    situation = entry.get("metadata", {}).get("situation", "inconnu")
    context = f"Genre: {genre}, Situation: {situation}"
    
    # Extraire le texte du dialogue
    dialogue_parts = []
    for msg in entry.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")[:300]  # Limiter à 300 chars
        if role == "system":
            dialogue_parts.append(f"[SCÈNE] {content}")
        elif role == "user":
            dialogue_parts.append(f"[JOUEUR] {content}")
        elif role == "assistant":
            dialogue_parts.append(f"[PERSONNAGE] {content}")
    
    dialogue_text = "\n".join(dialogue_parts)
    return context, dialogue_text


def translate_entry(entry: dict, idx: int) -> dict:
    """Traduit une entrée via Llama 3.3 70B."""
    context, dialogue_text = build_prompt(entry)
    
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": dialogue_text}
                        ],
                        "max_tokens": 2048,
                        "temperature": 0.7,
                    }
                )
                
                if resp.status_code == 200:
                    result = resp.json()
                    translated_text = result["choices"][0]["message"]["content"].strip()
                    
                    # Nettoyer le JSON
                    if translated_text.startswith("```"):
                        lines = translated_text.split("\n")
                        for i, line in enumerate(lines):
                            if line.strip().startswith("```json"):
                                translated_text = "\n".join(lines[i+1:])
                                break
                        if translated_text.endswith("```"):
                            translated_text = "\n".join(translated_text.split("\n")[:-1])
                        translated_text = translated_text.strip()
                    
                    translated_entry = json.loads(translated_text)
                    translated_entry["_translation_status"] = "success"
                    return translated_entry
                else:
                    err = resp.json().get("error", {}).get("message", str(resp.status_code))
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
                    else:
                        print(f"   ❌ {idx}: API Error {resp.status_code}: {err[:80]}")
                        entry["_translation_status"] = f"error_{resp.status_code}"
                        return entry
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"   ❌ {idx}: {type(e).__name__}: {e}")
                entry["_translation_status"] = type(e).__name__
                return entry
    
    return entry


def main():
    print("🚀 Traduction Ren'Py → Français RP (Sync Threaded)")
    print("=" * 60)
    
    # Charger le corpus
    input_path = "data/renpy-corpus-final.jsonl"
    output_path = "data/renpy-corpus-fr.jsonl"
    
    if not Path(input_path).exists():
        print(f"❌ Fichier introuvable: {input_path}")
        sys.exit(1)
    
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    
    print(f"📂 {len(entries)} entrées à traduire")
    print(f"🤖 Modèle: {MODEL}")
    print(f"⚡ Workers: {MAX_WORKERS}")
    print(f"⏱️  Timeout: {TIMEOUT}s")
    print()
    
    # Traduire en batch avec threads
    success_count = 0
    fail_count = 0
    results = []
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(translate_entry, entry, i): i
            for i, entry in enumerate(entries)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results.append(result)
            
            if result.get("_translation_status") == "success":
                success_count += 1
            else:
                fail_count += 1
            
            elapsed = time.time() - start_time
            rate = len(results) / elapsed if elapsed > 0 else 0
            
            # Afficher progression tous les 10 entrées
            if len(results) % 10 == 0 or len(results) == len(entries):
                eta = (len(entries) - len(results)) / rate if rate > 0 else 0
                print(f"   📊 {len(results)}/{len(entries)} ({len(results)/len(entries)*100:.0f}%) — "
                      f"✅ {success_count} ✅ ❌ {fail_count} — "
                      f"{rate:.1f} entrées/s — ETA: {eta:.0f}s")
            
            time.sleep(BATCH_DELAY)  # Rate limiting
    
    # Sauvegarder
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("✅ Traduction terminée!")
    print(f"   Temps total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"   Succès: {success_count}/{len(entries)} ({success_count/len(entries)*100:.0f}%)")
    print(f"   Échecs: {fail_count}/{len(entries)}")
    print(f"   Débit moyen: {len(entries)/total_time:.1f} entrées/s")
    print(f"   Sortie: {output_path}")


if __name__ == "__main__":
    main()
