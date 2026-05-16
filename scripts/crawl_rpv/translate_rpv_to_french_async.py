#!/usr/bin/env python3
"""
Traduction et adaptation RP de corpus Ren'Py en français — version async optimisée

Traduit les dialogues Ren'Py anglais vers un français de jeu de rôle naturel.
Utilise Llama 3.3 70B via Together AI.

Usage:
    python scripts/crawl_rpv/translate_rpv_to_french_async.py \
        --input data/renpy-corpus-final.jsonl \
        --output data/renpy-corpus-fr.jsonl
"""

import json
import os
import sys
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Auto-load .env
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

import httpx

# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
API_KEY = os.getenv("TOGETHER_API_KEY", "")
BATCH_DELAY = 0.5  # secondes entre les requêtes
CONCURRENCY = 8  # Requêtes parallèles maximales
MAX_RETRIES = 2
TIMEOUT = 120  # secondes par requête

# ============================================================
# PROMPT SYSTEM
# ============================================================

SYSTEM_PROMPT = """Tu es un traducteur spécialisé en jeux de rôle textuels français.

TÂCHE : Traduire et adapter un dialogue de Visual Novel (Ren'Py) de l'anglais vers un français de jeu de rôle naturel.

RÈGLES :
1. Registre RP : Utilise des tournures naturelles de dialogue français.
2. Ponctuation RP : Tiret français (-) pour les dialogues courts, guillemets français (« ») pour les longues phrases.
3. Idiomes : Remplace les expressions anglaises par leurs équivalents français naturels.
4. Ton : Garde le registre émotionnel (doux, dramatique, humoristique, effrayé...).
5. Noms de personnages : Conserve les noms anonymisés (Personnage1, Personnage2...).
6. Structure : Conserve la structure dialogue par dialogue.

CONTEXTE : {context}

FORMAT DE SORTIE : Renvoie UNIQUEMENT le JSON traduit, sans markdown ni texte supplémentaire.
Commence par {{ et termine par }}."""


def build_prompt(entry: dict) -> tuple[str, str]:
    """Construit le prompt pour une entrée."""
    genre = entry.get("metadata", {}).get("genre", "inconnu")
    situation = entry.get("metadata", {}).get("situation", "inconnu")
    context = f"Genre: {genre}, Situation: {situation}"
    
    # Extraire le texte du dialogue
    dialogue_parts = []
    for msg in entry.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")[:500]  # Limiter la longueur
        if role == "system":
            dialogue_parts.append(f"[SCÈNE] {content}")
        elif role == "user":
            dialogue_parts.append(f"[JOUEUR] {content}")
        elif role == "assistant":
            dialogue_parts.append(f"[PERSONNAGE] {content}")
    
    dialogue_text = "\n".join(dialogue_parts)
    
    system = SYSTEM_PROMPT.format(context=context)
    
    user = f"""Traduire en français RP :
--- DIALOGUE ORIGINAL ---
{dialogue_text}
--- FIN ---
Renvoie UNIQUEMENT le JSON traduit avec la même structure."""
    
    return system, user


async def translate_entry(
    client: httpx.AsyncClient,
    entry: dict,
    session_id: int
) -> dict:
    """Traduit une entrée via Llama 3.3 70B."""
    system_msg, user_msg = build_prompt(entry)
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
                timeout=TIMEOUT,
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
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"   ❌ Session {session_id}: API Error {resp.status_code}: {err[:80]}")
                    entry["_translation_status"] = f"error_{resp.status_code}"
                    return entry
                    
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"   ❌ Session {session_id}: {type(e).__name__}: {e}")
                entry["_translation_status"] = type(e).__name__
                return entry
    
    return entry


async def main():
    print("🚀 Traduction Ren'Py → Français RP (Async)")
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
    print(f"⚡ Concurrency: {CONCURRENCY}")
    print(f"⏱️  Timeout: {TIMEOUT}s")
    print()
    
    # Traduire en batch async
    success_count = 0
    fail_count = 0
    results = []
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    async def translate_with_semaphore(entry, idx):
        async with semaphore:
            return await translate_entry(client, entry, idx)
    
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = [
            translate_with_semaphore(entry, i)
            for i, entry in enumerate(entries)
        ]
        
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            result = await task
            results.append(result)
            
            if result.get("_translation_status") == "success":
                success_count += 1
            else:
                fail_count += 1
            
            # Progression tous les 10 entrées
            if i % 10 == 0 or i == len(entries):
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(entries) - i) / rate if rate > 0 else 0
                print(f"   📊 Progression: {i}/{len(entries)} ({i/len(entries)*100:.0f}%) — "
                      f"✅ {success_count} ✅ ❌ {fail_count} — "
                      f"{rate:.1f} entrées/s — ETA: {eta:.0f}s")
    
    # Sauvegarder
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("✅ Traduction terminée!")
    print(f"   Temps total: {total_time:.0f}s")
    print(f"   Succès: {success_count}/{len(entries)} ({success_count/len(entries)*100:.0f}%)")
    print(f"   Échecs: {fail_count}/{len(entries)}")
    print(f"   Débit moyen: {len(entries)/total_time:.1f} entrées/s")
    print(f"   Sortie: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
