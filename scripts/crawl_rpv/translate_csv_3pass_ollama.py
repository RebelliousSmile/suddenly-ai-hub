#!/usr/bin/env python3
"""
Ren'Py Translation Pipeline (3 passes) — Ollama Local with retries
"""
import csv
import os
import time
import sys
import json
import re
import httpx

# --- Configuration ---
INPUT_FILE = "data/renpy-corpus-flat.csv"
OUT_P1 = "data/renpy-corpus-flat-p1.csv"
OUT_P2 = "data/renpy-corpus-flat-p2.csv"
OUT_P3 = "data/renpy-corpus-flat-fr.csv"

OLLAMA_URL = "http://localhost:11434"
BATCH_SIZE = 5  # Plus petit pour être stable
DELAY = 3.0  # Plus long pour ne pas saturer
MODEL_P1 = "mistral"
MODEL_P2 = "mistral"
MODEL_P3 = "mistral"
TEMPERATURE = 0.7
MAX_RETRIES = 3

def ollama_check():
    """Check if Ollama is alive."""
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False

def ollama_complete(messages, model, max_tokens=2048, retries=MAX_RETRIES):
    """Send request to local Ollama with retries."""
    for attempt in range(retries):
        try:
            response = httpx.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_predict": max_tokens,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                print(f"    ⚠️ Retry {attempt+1}/{retries}...")
                time.sleep(5 * (attempt + 1))
            else:
                return None

def translate_batch(batch, system_prompt, model):
    """Translate a batch of lines."""
    if not batch: return []
    
    user_prompt = "Translate the following lines to French. Output ONLY the translations, one per line, matching the order of the input.\n\n"
    for i, row in enumerate(batch):
        user_prompt += f"{i+1}. {row['content']}\n"
    user_prompt += "\nTranslations:"
    
    response = ollama_complete(
        [{"role": "system", "content": system_prompt}, 
         {"role": "user", "content": user_prompt}],
        model,
        2048
    )
    
    if response is None:
        return [row['content'] for row in batch]
    
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    
    if len(lines) != len(batch):
        print(f"    ⚠️ Count mismatch: got {len(lines)}, expected {len(batch)}. Taking all lines.")
    
    return lines

def main():
    print("🔄 Starting 3-Pass CSV Translation Pipeline (Ollama Local)")
    print("=" * 60)
    
    # Check Ollama first
    print("🔍 Checking Ollama...")
    if not ollama_check():
        print("❌ Ollama is not running! Start it with: ollama serve")
        return
    print("✅ Ollama is alive!")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"📊 Total entries: {len(rows)}")
    
    # --- PASS 1: Brute ---
    print("\n📦 PASS 1 — Brute Translation...")
    sys_prompt_p1 = "You are a translator. Translate English to French. Be literal but natural."
    p1_rows = []
    
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        new_lines = translate_batch(batch, sys_prompt_p1, MODEL_P1)
        for row, text in zip(batch, new_lines):
            row['content_fr'] = text
        p1_rows.append(row)
        
        if (i // BATCH_SIZE) % 50 == 0:
            print(f"   → {i}/{len(rows)}")
        
        if i + BATCH_SIZE < len(rows):
            time.sleep(DELAY)
            
    # Save P1
    with open(OUT_P1, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['genre', 'situation', 'role', 'content', 'content_fr'])
        writer.writeheader()
        writer.writerows(p1_rows)
    print(f"   ✅ P1 saved ({len(p1_rows)} lines)")

    # --- PASS 2: Style ---
    print("\n🎨 PASS 2 — RP Style Refinement...")
    sys_prompt_p2 = "You are an editor for French text RPGs. Polish the provided French translation to make it more natural, idiomatic, and fitting for a RP context. Keep the meaning exactly the same."
    p2_rows = []
    
    for i in range(0, len(p1_rows), BATCH_SIZE):
        batch = p1_rows[i:i+BATCH_SIZE]
        user_prompt = "Polish the following French sentences for a French RP context. Output ONLY the polished sentences, one per line.\n\n"
        for j, row in enumerate(batch):
            user_prompt += f"{j+1}. {row['content_fr']}\n"
        user_prompt += "\nPolished versions:"
        
        response = ollama_complete(
            [{"role": "system", "content": sys_prompt_p2}, 
             {"role": "user", "content": user_prompt}],
            MODEL_P2,
            2048
        )
        
        if response:
            new_lines = [l.strip() for l in response.split('\n') if l.strip()]
            for row, text in zip(batch, new_lines):
                row['content_fr'] = text
        
        p2_rows.extend(batch)
        
        if (i // BATCH_SIZE) % 50 == 0:
            print(f"   → {i}/{len(p1_rows)}")
            
        if i + BATCH_SIZE < len(p1_rows):
            time.sleep(DELAY)
            
    # Save P2
    with open(OUT_P2, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['genre', 'situation', 'role', 'content', 'content_fr'])
        writer.writeheader()
        writer.writerows(p2_rows)
    print(f"   ✅ P2 saved ({len(p2_rows)} lines)")

    # --- PASS 3: Cleanup ---
    print("\n🛠️ PASS 3 — Cleanup & Validation...")
    final_rows = []
    for row in p2_rows:
        fr_text = row['content_fr']
        fr_text = fr_text.strip()
        fr_text = re.sub(r'^```.*?\n(.*?)\n```$', r'\1', fr_text, flags=re.DOTALL)
        row['content_fr'] = fr_text
        final_rows.append(row)
        
    with open(OUT_P3, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['genre', 'situation', 'role', 'content', 'content_fr'])
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"   ✅ P3 (Final) saved ({len(final_rows)} lines)")
    print("\n🎉 Pipeline complete!")

if __name__ == "__main__":
    main()
