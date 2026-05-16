#!/usr/bin/env python3
"""
Traduction et adaptation RP de corpus Ren'Py en français

Traduit les dialogues Ren'Py anglais vers un français de jeu de rôle naturel.
Utilise Mistral Large via Together AI pour la qualité.

Usage:
    python scripts/crawl_rpv/translate_rpv_to_french.py \
        --input data/renpy-corpus-final.jsonl \
        --output data/renpy-corpus-fr.jsonl \
        --batch-size 10
"""

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Auto-load .env
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=True)

import httpx

# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
BATCH_SIZE = 10  # Conversations par batch API
MAX_RETRIES = 3
BATCH_DELAY = 1.0  # secondes entre les batches
THREADS = 5  # Threads parallèles pour le batch

# ============================================================
# PROMPT SYSTEM — Spécialisé pour la traduction RP
# ============================================================

TRANSLATION_SYSTEM_PROMPT = """Tu es un traducteur spécialisé en jeux de rôle textuels français.

TÂCHE : Traduire et adapter un dialogue de Visual Novel (Ren'Py) de l'anglais vers un français de jeu de rôle naturel.

RÈGLES D'ADAPTATION :
1. **Registre RP** : Utilise des tournures naturelles de dialogue français (ex: « Je ne sais pas » au lieu de « I don't know »).
2. **Ponctuation RP** : Utilise le tiret français (-) pour les dialogues courts, et les guillemets français (« ») pour les longues phrases.
3. **Idiomes** : Remplace les expressions anglaises par leurs équivalents français naturels (ex: "Hang in there" → « Tiens bon »).
4. **Ton** : Garde le registre émotionnel (doux, dramatique, humoristique, effrayé...). Le ton est aussi important que le sens.
5. **Noms de personnages** : Conserve les noms anonymisés (Personnage1, Personnage2...). Si des noms propres existent, garde-les.
6. **Structure** : Conserve la structure dialogue par dialogue. Ne fusionne pas les tours de parole.

CONTEXTE fourni :
- Genre : {genre}
- Situation : {situation}

Adapte le style en conséquence (ex: horreur → ton angoissé, romance → ton tendre, instruction → ton neutre).

FORMAT DE SORTIE : 
Renvoie UNIQUEMENT le JSON traduit, sans markdown ni texte supplémentaire. Ne modifie PAS les noms des champs JSON."""


def build_user_prompt(entry: dict) -> str:
    """Construit le prompt utilisateur pour une entrée."""
    genre = entry.get("metadata", {}).get("genre", "inconnu")
    situation = entry.get("metadata", {}).get("situation", "inconnu")
    
    # Construire le texte du dialogue à traduire
    dialogue_parts = []
    for msg in entry.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            dialogue_parts.append(f"[SCÈNE] {content}")
        elif role == "user":
            dialogue_parts.append(f"[JOUEUR] {content}")
        elif role == "assistant":
            dialogue_parts.append(f"[PERSONNAGE] {content}")
    
    dialogue_text = "\n".join(dialogue_parts)
    
    prompt = f"""Traduire en français de jeu de rôle le dialogue suivant :

--- DIALOGUE ORIGINAL ---
{dialogue_text}
--- FIN DU DIALOGUE ---

Adapte le style pour correspondre à :
- Genre : {genre}
- Situation : {situation}

IMPORTANT : Renvoie UNIQUEMENT le JSON traduit avec la même structure."""
    
    return prompt


def translate_entry(entry: dict) -> dict:
    """Traduit une seule entrée en français via Mistral Large."""
    prompt = build_user_prompt(entry)
    
    # Nettoyer le JSON original pour la structure de sortie
    original_json = json.dumps(entry, ensure_ascii=False)
    
    # Prompt avec exemple de structure attendue
    full_prompt = f"""{TRANSLATION_SYSTEM_PROMPT.format(genre=entry.get("metadata", {}).get("genre", "inconnu"), situation=entry.get("metadata", {}).get("situation", "inconnu"))}

STRUCTURE JSON attendue (ne pas modifier les noms de champs) :
{{
    "messages": [
        {{
            "role": "system",
            "content": "Le contexte de scène traduit en français"
        }},
        {{
            "role": "user",
            "content": "Le texte du joueur traduit en français"
        }},
        {{
            "role": "assistant",
            "content": "La réponse du personnage traduite en français"
        }}
    ],
    "metadata": {{
        "source": "renpy",
        "genre": "{entry.get('metadata', {}).get('genre', 'inconnu')}",
        "situation": "{entry.get('metadata', {}).get('situation', 'inconnu')}",
        "label": "{entry.get('metadata', {}).get('label', '')}",
        "dialogues_count": {entry.get('metadata', {}).get('dialogues_count', 0)},
        "characters": {json.dumps(entry.get('metadata', {}).get('characters', []))},
        "original_language": "en",
        "translated_to": "fr"
    }}
}}

DIALOGUE ORIGINAL À TRADUIRE :
{prompt}

Renvoie UNIQUEMENT le JSON traduit. Commence par {{ et termine par }}."""
    
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "user", "content": full_prompt}
                        ],
                        "max_tokens": 2048,
                        "temperature": 0.7,  # Un peu de créativité pour l'adaptation
                        "top_p": 0.9,
                    }
                )
                response.raise_for_status()
                result = response.json()
                translated_text = result["choices"][0]["message"]["content"].strip()
                
                # Nettoyer le JSON de la réponse (peut contenir des marqueurs markdown)
                translated_text = translated_text.strip()
                if translated_text.startswith("```"):
                    lines = translated_text.split("\n")
                    for i, line in enumerate(lines):
                        if line.strip().startswith("```json"):
                            translated_text = "\n".join(lines[i+1:])
                            break
                    # Retirer les ``` de fin
                    if translated_text.endswith("```"):
                        translated_text = "\n".join(translated_text.split("\n")[:-1])
                    translated_text = translated_text.strip()
                
                # Parser et retourner
                translated_entry = json.loads(translated_text)
                
                # Vérifier que la structure est correcte
                if "messages" not in translated_entry:
                    print(f"   ⚠️  Structure JSON invalide, retour au texte brut")
                    translated_entry = entry.copy()
                    translated_entry["_translation_failed"] = True
                    translated_entry["_translation_text"] = translated_text
                
                return translated_entry
                
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"   ⚠️  Échec traduction (tentative {attempt+1}/{MAX_RETRIES}): {type(e).__name__}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Backoff exponentiel
            else:
                print(f"   ❌ Échec final, entrée conservée en anglais")
                entry["_translation_failed"] = True
                entry["_translation_error"] = str(e)
                return entry
    
    return entry


def main():
    print("🚀 Traduction Ren'Py → Français RP")
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
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔁 Max retries: {MAX_RETRIES}")
    print()
    
    # Traduire par batch
    results = []
    success_count = 0
    fail_count = 0
    
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(entries) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"📦 Batch {batch_num}/{total_batches} ({len(batch)} entrées)...")
        
        # Traduire en parallèle (threads)
        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = {
                executor.submit(translate_entry, entry): idx
                for idx, entry in enumerate(batch)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                if result.get("_translation_failed"):
                    fail_count += 1
                else:
                    success_count += 1
                results.append(result)
        
        time.sleep(BATCH_DELAY)  # Rate limiting API
    
    # Sauvegarder le résultat
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print("✅ Traduction terminée!")
    print(f"   Succès: {success_count}")
    print(f"   Échecs: {fail_count}")
    print(f"   Sortie: {output_path}")


if __name__ == "__main__":
    main()
