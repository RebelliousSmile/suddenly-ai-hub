#!/usr/bin/env python3
"""
Pipeline complet de traduction RP (3 passes) sur le corpus Ren'Py.
Version corrigée : moins de workers, plus de delays, fallback robuste.

Usage:
    python scripts/crawl_rpv/translate_full_pipeline_v2.py
"""

import json
import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import httpx

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
API_KEY = os.getenv("TOGETHER_API_KEY", "")
MAX_WORKERS = 3  # Réduit de 8 à 3 pour éviter les 429
TIMEOUT = 300
BATCH_DELAY = 2.0  # Augmenté de 0.5 à 2s

SYSTEM_P1 = "Tu es un traducteur spécialisé en jeux de rôle textuels français. Traduire un dialogue de Visual Novel (Ren'Py) de l'anglais vers un français de jeu de rôle naturel. Utilise le tiret français (-) et les guillemets français (« »). Conserve la structure JSON. Pas de commentaire ni markdown. Commence par {{ et termine par }}."
SYSTEM_P2 = "Tu es un éditeur de littérature RP française. Tu reçois une traduction brute. 1. Guillemets anglais → « ». 2. Tirets longs — → tiret -. 3. Adapter les expressions idiomatiques au français courant RP. 4. Conserver le ton. 5. UNIQUEMENT le JSON valide, sans markdown."

INPUT = "data/renpy-corpus-final.jsonl"
OUT_P1 = "data/renpy-corpus-p1.jsonl"
OUT_P2 = "data/renpy-corpus-p2.jsonl"
OUT_FR = "data/renpy-corpus-fr.jsonl"


def build_prompt(entry):
    parts = []
    for msg in entry.get("messages", []):
        role, content = msg.get("role", "user"), msg.get("content", "")[:300]
        tag = {"system": "[SCÈNE]", "user": "[JOUEUR]", "assistant": "[PERSONNAGE]"}.get(role, role.upper())
        parts.append(f"{tag} {content}")
    return "\n".join(parts)


def call_together(client, prompt, system, max_tokens=2048, retries=3):
    """Appel API avec retry et backoff."""
    for attempt in range(retries):
        try:
            r = client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role":"system","content":system}, {"role":"user","content":prompt}],
                      "max_tokens": max_tokens, "temperature": 0.7},
                timeout=TIMEOUT,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            elif r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"   ⏳ Rate limit 429, attente {wait}s...")
                time.sleep(wait)
            else:
                return f"ERR_{r.status_code}"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"EXCEPT_{type(e).__name__}"
    return "RETRY_FAILED"


def pass1(entry):
    prompt = build_prompt(entry)
    with httpx.Client() as client:
        text = call_together(client, prompt, SYSTEM_P1)
    entry["_p1"] = text if text and not text.startswith("ERR") else ""
    entry["_status_p1"] = "ok" if text and not text.startswith("ERR") else text[:50]
    return entry


def pass2(entry):
    text = entry.get("_p1", "")
    if not text or text.startswith("ERR"):
        entry["_status_p2"] = "missing_p1"
        entry["_p2"] = ""
        return entry
    with httpx.Client() as client:
        result = call_together(client, text, SYSTEM_P2)
    if result and not result.startswith("ERR"):
        entry["_p2"] = result.strip().strip("`").strip("json").strip()
        entry["_status_p2"] = "ok"
    else:
        entry["_p2"] = text  # fallback: garder la traduction brute
        entry["_status_p2"] = f"p2_{result[:30]}"
    return entry


def pass3(entry):
    text = entry.get("_p2", "")
    if not text:
        entry["_status_p3"] = "no_data"
        entry["_final"] = {}
        return entry

    # Nettoyer les guillemets
    cleaned = text
    cleaned = re.sub(r'""([^""]+)""', r'« \1 »', cleaned)
    cleaned = re.sub(r'“([^“”]+)”', r'« \1 »', cleaned)
    # Normaliser les apostrophes et tirets
    cleaned = re.sub(r"(?<!\S)'", "’", cleaned)

    # Validation JSON
    try:
        data = json.loads(cleaned)
        entry["_final"] = data
        entry["_status_p3"] = "ok"
    except json.JSONDecodeError:
        # Fallback : extraire le JSON du texte brut
        # Chercher le premier { et le dernier }
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start:end])
                entry["_final"] = data
                entry["_status_p3"] = "extracted"
            except json.JSONDecodeError:
                entry["_status_p3"] = "fail"
                # Garder le texte brut en fallback
                entry["_final"] = {"_raw_translation": cleaned[:500], "_status": "translation_failed"}
        else:
            entry["_status_p3"] = "no_json"
            entry["_final"] = {"_raw_translation": cleaned[:500], "_status": "no_json_found"}

    return entry


def main():
    print("🚀 PIPELINE TRADUCTION RP v2 (3 passes, 3 workers)")
    print("=" * 60)

    entries = []
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"📂 {len(entries)} entrées chargées\n")

    # --- PASS 1 ---
    print("📦 PASS 1 — Traduction brute...")
    p1_results = []
    start_p1 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(pass1, e): i for i, e in enumerate(entries)}
        for f in as_completed(futures):
            p1_results.append(f.result())
            n = len(p1_results)
            ok = sum(1 for e in p1_results if e.get("_status_p1") == "ok")
            print(f"   → {n}/{len(entries)} (OK: {ok})")
            time.sleep(BATCH_DELAY)
    with open(OUT_P1, "w", encoding="utf-8") as f:
        for e in p1_results:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")
    print(f"   ✅ P1 sauvegardé ({time.time()-start_p1:.0f}s)\n")

    # --- PASS 2 ---
    print("🎨 PASS 2 — Raffinement Style RP...")
    p2_results = []
    start_p2 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(pass2, e): i for i, e in enumerate(p1_results)}
        for f in as_completed(futures):
            p2_results.append(f.result())
            n = len(p2_results)
            ok = sum(1 for e in p2_results if e.get("_status_p2") == "ok")
            print(f"   → {n}/{len(entries)} (OK: {ok})")
            time.sleep(BATCH_DELAY)
    with open(OUT_P2, "w", encoding="utf-8") as f:
        for e in p2_results:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")
    print(f"   ✅ P2 sauvegardé ({time.time()-start_p2:.0f}s)\n")

    # --- PASS 3 ---
    print("🛠️ PASS 3 — Nettoyage + Validation JSON...")
    final_results = []
    ok_count = 0
    extracted_count = 0
    fail_count = 0
    for e in p2_results:
        res = pass3(e)
        if res.get("_status_p3") == "ok":
            ok_count += 1
        elif res.get("_status_p3") == "extracted":
            extracted_count += 1
        else:
            fail_count += 1
        final_results.append(res)

    with open(OUT_FR, "w", encoding="utf-8") as f:
        for e in final_results:
            json.dump(e.get("_final", {}), f, ensure_ascii=False, indent=2)
            f.write("\n")

    total_time = time.time() - start_p1
    print(f"\n{'='*60}")
    print("✅ PIPELINE TERMINÉ !")
    print(f"   Temps total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"   ✅ JSON valide: {ok_count}")
    print(f"   ⚠️  JSON extrait: {extracted_count}")
    print(f"   ❌ Échec: {fail_count}")
    print(f"   Taux de succès: {(ok_count+extracted_count)/len(entries)*100:.0f}%")
    print(f"   Sortie : {OUT_FR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
