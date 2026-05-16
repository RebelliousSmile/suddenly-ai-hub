#!/usr/bin/env python3
"""
Translation Script — Ren'Py Corpus CSV (EN -> FR)

Reads a flat CSV with columns: genre, situation, role, content
Outputs a new CSV with an additional column: content_fr
"""

import csv
import os
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
from together import Together

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# --- Configuration ---
INPUT_FILE = "data/renpy-corpus-flat.csv"
OUTPUT_FILE = "data/renpy-corpus-translated.csv"
BATCH_SIZE = 5  # Translate 5 lines at a time
DELAY_BETWEEN_BATCHES = 2.0  # seconds to avoid rate limits
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

def translate_batch(lines_to_translate):
    """Translate a batch of lines to French."""
    if not lines_to_translate:
        return []
    
    # Build the prompt for all lines in this batch
    prompt_parts = []
    for i, line in enumerate(lines_to_translate):
        content = line['content']
        prompt_parts.append(f"Line {i+1}: {content}")
    
    system_prompt = (
        "You are a professional EN→FR translator specializing in visual novel dialogue.\n"
        "Translate ONLY the English text to natural French. Preserve the tone and register.\n"
        "Output ONLY the translated text, nothing else. No markdown, no explanations.\n"
        "Output format: one translated line per original line, in order."
    )
    
    user_prompt = "\n".join(prompt_parts) + "\n\nTranslate each line above to French:"
    
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Low temperature for consistent translation
            max_tokens=200,
        )
        result = response.choices[0].message.content.strip()
        return [line.strip() for line in result.split('\n')]
    except Exception as e:
        print(f"    ❌ Batch failed: {e}")
        return [line['content'] for line in lines_to_translate]  # Fallback to original

def main():
    print("🔄 Translation: Ren'Py Corpus (EN → FR)")
    print("=" * 50)
    
    # Read input CSV
    rows = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"📊 Total lines to translate: {len(rows)}")
    
    # Process in batches
    translated_rows = []
    total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(rows))
        batch = rows[start_idx:end_idx]
        
        print(f"\n📤 Batch {batch_num + 1}/{total_batches} "
              f"(lines {start_idx + 1}-{end_idx})")
        
        translations = translate_batch(batch)
        
        for i, (original, translated) in enumerate(zip(batch, translations)):
            translated_row = original.copy()
            translated_row['content_fr'] = translated
            translated_rows.append(translated_row)
        
        # Save progress after each batch
        with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, 
                fieldnames=['genre', 'situation', 'role', 'content', 'content_fr'])
            writer.writeheader()
            writer.writerows(translated_rows)
        
        print(f"    ✅ {len(translations)} lines translated")
        print(f"    💾 Progress saved: {len(translated_rows)}/{len(rows)}")
        
        if batch_num < total_batches - 1:
            print(f"    ⏳ Delaying {DELAY_BETWEEN_BATCHES}s before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES)
    
    print("\n🎉 Translation complete!")
    print(f"    Output: {OUTPUT_FILE}")
    print(f"    Total: {len(translated_rows)} lines translated")

if __name__ == "__main__":
    main()
