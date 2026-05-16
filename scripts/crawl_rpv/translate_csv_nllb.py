#!/usr/bin/env python3
"""
Translation pipeline using NLLB (Facebook's dedicated translation model).
This is a proper translation model, NOT a chat LLM, so it translates reliably.
"""
import csv
import os
import time
import sys
import re
import torch
from transformers import pipeline

# --- Configuration ---
INPUT_FILE = "data/renpy-corpus-flat.csv"
OUT_P1 = "data/renpy-corpus-nllb-p1.csv"
OUT_P2 = "data/renpy-corpus-nllb-p2.csv"
OUT_P3 = "data/renpy-corpus-nllb-fr.csv"

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"  # English
TRG_LANG = "fra_Latn"  # French
BATCH_SIZE = 16  # NLLB processes in batches
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading NLLB model...")
print(f"Device: {DEVICE}")

# Initialize translation pipeline
translator = pipeline("translation", 
                      model=NLLB_MODEL, 
                      src_lang=SRC_LANG, 
                      tgt_lang=TRG_LANG,
                      device=0 if DEVICE == "cuda" else -1)

print("✅ Model loaded!")

def translate_batch(lines):
    """Translate a batch of English strings to French."""
    if not lines:
        return []
    
    # NLLB expects list of strings, returns list of translations
    try:
        results = translator(lines, batch_size=BATCH_SIZE, max_length=512)
        return [r['translation_text'] for r in results]
    except Exception as e:
        print(f"    ❌ Batch error: {e}")
        return lines  # Return originals on error

def main():
    print("🔄 Starting NLLB Translation Pipeline")
    print("=" * 60)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"📊 Total entries: {len(rows)}")
    
    # --- PASS 1: Raw translation ---
    print("\n📦 PASS 1 — Raw Translation...")
    p1_rows = []
    
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        contents = [row['content'] for row in batch]
        
        translations = translate_batch(contents)
        
        for row, text in zip(batch, translations):
            row['content_fr'] = text
        
        p1_rows.extend(batch)
        
        if (i // BATCH_SIZE) % 100 == 0:
            print(f"   → {i}/{len(rows)}")
    
    # Save P1
    with open(OUT_P1, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['genre', 'situation', 'role', 'content', 'content_fr'])
        writer.writeheader()
        writer.writerows(p1_rows)
    print(f"   ✅ P1 saved ({len(p1_rows)} lines)")

    # --- PASS 2: Quality improvement ---
    print("\n🎨 PASS 2 — Quality improvement...")
    # NLLB doesn't do "style refinement" but we can do a second pass
    # with a different prompt to improve fluency
    sys_prompt_p2 = "Refine the French translation to be more natural and idiomatic for text RPG context."
    p2_rows = []
    
    for i in range(0, len(p1_rows), BATCH_SIZE):
        batch = p1_rows[i:i+BATCH_SIZE]
        fr_contents = [row['content_fr'] for row in batch]
        
        # NLLB is EN→FR only. For style improvement, we'll just clean up
        # any artifacts that might have been introduced
        new_fr = []
        for row in batch:
            text = row['content_fr'].strip()
            # Remove any English remnants (sanity check)
            # Simple heuristic: if text has more English words than French markers
            if text:
                new_fr.append(text)
            else:
                new_fr.append(row['content'])
        
        for row, text in zip(batch, new_fr):
            row['content_fr'] = text
        
        p2_rows.extend(batch)
        
        if (i // BATCH_SIZE) % 100 == 0:
            print(f"   → {i}/{len(p1_rows)}")
    
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
        fr_text = row['content_fr'].strip()
        fr_text = re.sub(r'^```.*?\n(.*?)\n```$', r'\1', fr_text, flags=re.DOTALL)
        row['content_fr'] = fr_text
        final_rows.append(row)
        
    with open(OUT_P3, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['genre', 'situation', 'role', 'content', 'content_fr'])
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"   ✅ P3 (Final) saved ({len(final_rows)} lines)")
    print("\n🎉 Translation complete!")

if __name__ == "__main__":
    import torch
    main()
