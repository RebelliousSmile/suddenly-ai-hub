#!/usr/bin/env python3
"""
Ren'Py Translation Pipeline (3 passes) for flat CSV.
"""
import csv
import os
import time
import sys
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from together import Together

# --- Setup ---
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

INPUT_FILE = "data/renpy-corpus-flat.csv"
OUT_P1 = "data/renpy-corpus-flat-p1.csv"
OUT_P2 = "data/renpy-corpus-flat-p2.csv"
OUT_P3 = "data/renpy-corpus-flat-fr.csv"

BATCH_SIZE = 5  # 5 lignes par batch pour garder un peu de contexte
DELAY = 2.0
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

def translate_batch(batch, system_prompt):
    """Translate a batch of lines."""
    if not batch: return []
    
    user_prompt = "Translate the following lines to French. Output ONLY the translations, one per line, matching the order of the input.\n\n"
    for i, row in enumerate(batch):
        user_prompt += f"{i+1}. {row['content']}\n"
    user_prompt += "\nTranslations:"
    
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}, 
                      {"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        result = response.choices[0].message.content.strip()
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        
        # Safety check: ensure we get back the expected number of lines
        if len(lines) != len(batch):
            # Fallback: assume single line per input was misunderstood
            print(f"    ⚠️ Batch output count mismatch: got {len(lines)}, expected {len(batch)}. Taking all lines.")
            
        return lines
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return [row['content'] for row in batch] # Return originals on failure

def main():
    print("🔄 Starting 3-Pass CSV Translation Pipeline")
    print("=" * 50)
    
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
        new_lines = translate_batch(batch, sys_prompt_p1)
        for row, text in zip(batch, new_lines):
            row['content_fr'] = text
        p1_rows.append(row)
        
        if (i // BATCH_SIZE) % 200 == 0:
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
        # Reconstruct the prompt with current FR text
        user_prompt = "Polish the following French sentences for a French RP context. Output ONLY the polished sentences, one per line.\n\n"
        for j, row in enumerate(batch):
            user_prompt += f"{j+1}. {row['content_fr']}\n"
        user_prompt += "\nPolished versions:"
        
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": sys_prompt_p2}, 
                          {"role": "user", "content": user_prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            result = response.choices[0].message.content.strip()
            new_lines = [l.strip() for l in result.split('\n') if l.strip()]
            
            for row, text in zip(batch, new_lines):
                row['content_fr'] = text
            
            p2_rows.extend(batch)
        except Exception as e:
            print(f"    ❌ Pass 2 Error: {e}")
            p2_rows.extend(batch)
            
        if (i // BATCH_SIZE) % 200 == 0:
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
    # We can just pass through the cleaned data or apply a regex cleanup
    # Since it's text, we rely on the API to have done the work, but let's clean markdown if any
    final_rows = []
    for row in p2_rows:
        fr_text = row['content_fr']
        fr_text = fr_text.strip()
        fr_text = re.sub(r'^```.*?\n(.*?)\n```$', r'\1', fr_text, flags=re.DOTALL)
        row['content_fr'] = fr_text
        final_rows.append(row)
        
    # Save P3
    with open(OUT_P3, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['genre', 'situation', 'role', 'content', 'content_fr'])
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"   ✅ P3 (Final) saved ({len(final_rows)} lines)")
    print("\n🎉 Pipeline complete!")

if __name__ == "__main__":
    main()
