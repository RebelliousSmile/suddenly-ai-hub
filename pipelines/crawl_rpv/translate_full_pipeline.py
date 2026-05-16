#!/usr/bin/env python3
"""
Pipeline complet de traduction RP (3 passes) sur le corpus Ren'Py.
Pass 1: Traduction brute
Pass 2: Raffinement style RP
Pass 3: Nettoyage Python + Validation JSON
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
MAX_WORKERS = 8
TIMEOUT = 300
BATCH_DELAY = 0.5

SYSTEM_P1 = "Tu es un traducteur spécialisé en jeux de rôle textuels français. Traduire et adapter un dialogue de Visual Novel (Ren'Py) de l'anglais vers un français de jeu de rôle naturel. Utilise le tiret français (-) et les guillemets français (« »). Conserve la structure JSON originale. Ne fournis aucun commentaire ni markdown. Commence toujours par {{ et termine par }}."
SYSTEM_P2 = "Tu es un éditeur de littérature RP française. Tu reçois une traduction brute. TÂCHES : 1. Remplacer les guillemets anglais \"\" par « ». 2. Remplacer les tirets longs — par le tiret français -. 3. Adapter les expressions idiomatiques au français courant RP. 4. Conserver le ton (doux, dramatique, technique...). 5. Renvoyer UNIQUEMENT le JSON valide, sans markdown. Ne change PAS les noms de personnages."

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

def pass1(entry):
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.post("https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role":"system","content":SYSTEM_P1}, {"role":"user","content":build_prompt(entry)}], "max_tokens": 2048, "temperature": 0.7})
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"].strip().strip("`").strip("json").strip()
                entry["_p1"] = text
                entry["_status_p1"] = "ok"
                return entry
            else:
                entry["_status_p1"] = f"err_{r.status_code}"
                return entry
    except Exception as e:
        entry["_status_p1"] = str(e)
        return entry

def pass2(entry):
    text = entry.get("_p1", "")
    if not text:
        entry["_status_p2"] = "missing_p1"
        return entry
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.post("https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role":"system","content":SYSTEM_P2}, {"role":"user","content":text}], "max_tokens": 2048, "temperature": 0.7})
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"].strip().strip("`").strip("json").strip()
                entry["_p2"] = text
                entry["_status_p2"] = "ok"
                return entry
            else:
                entry["_status_p2"] = f"err_{r.status_code}"
                return entry
    except Exception as e:
        entry["_status_p2"] = str(e)
        return entry

def pass3(entry):
    text = entry.get("_p2", "")
    if not text:
        entry["_status_p3"] = "missing_p2"
        return entry
    
    # Nettoyage
    cleaned = re.sub(r'""([^""]+)""', r'« \1 »', text)
    cleaned = re.sub(r'“([^“”]+)”', r'« \1 »', cleaned)
    cleaned = cleaned.replace("— ", "- ").replace(" — ", " - ")
    
    # Validation
    try:
        data = json.loads(cleaned)
        entry["_final"] = data
        entry["_status_p3"] = "ok"
    except json.JSONDecodeError:
        # Fallback: essayer de parser quand même si possible
        entry["_status_p3"] = "parse_fail"
        entry["_final"] = json.loads(text) if text else {}
    
    return entry

def main():
    print("🚀 LANCEMENT PIPELINE TRADUCTION RACE (386 entrées)")
    print("=" * 60)
    
    entries = []
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    
    print(f"📂 {len(entries)} entrées chargées\n")
    
    # --- PASS 1 ---
    print("📦 PASS 1 — Traduction brute...")
    p1_results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(pass1, e): i for i, e in enumerate(entries)}
        for f in as_completed(futures):
            p1_results.append(f.result())
            n = len(p1_results)
            if n % 50 == 0 or n == len(entries):
                print(f"   → {n}/{len(entries)}")
    with open(OUT_P1, "w", encoding="utf-8") as f:
        for e in p1_results:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")
    print(f"   ✅ Fichier P1 sauvegardé ({time.time()-start:.0f}s)\n")
    
    # --- PASS 2 ---
    print("🎨 PASS 2 — Raffinement Style RP...")
    p2_results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(pass2, e): i for i, e in enumerate(p1_results)}
        for f in as_completed(futures):
            p2_results.append(f.result())
            n = len(p2_results)
            if n % 50 == 0 or n == len(entries):
                print(f"   → {n}/{len(entries)}")
    with open(OUT_P2, "w", encoding="utf-8") as f:
        for e in p2_results:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")
    print(f"   ✅ Fichier P2 sauvegardé ({time.time()-start:.0f}s)\n")
    
    # --- PASS 3 ---
    print("🛠️ PASS 3 — Nettoyage + Validation JSON...")
    final_results = []
    ok_count = 0
    for e in p2_results:
        res = pass3(e)
        if res.get("_status_p3") == "ok":
            ok_count += 1
        final_results.append(res)
    
    # Sauvegarde finale
    with open(OUT_FR, "w", encoding="utf-8") as f:
        for e in final_results:
            json.dump(e.get("_final", {}), f, ensure_ascii=False, indent=2)
            f.write("\n")
    
    total_time = time.time() - start
    print(f"\n{'='*60}")
    print("✅ PIPELINE TERMINÉ !")
    print(f"   Temps total: {total_time:.0f}s")
    print(f"   Entrées finales : {ok_count}/{len(entries)}")
    print(f"   Sortie : {OUT_FR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
