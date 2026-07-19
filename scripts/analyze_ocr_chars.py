"""Analyze German/Latin character errors in OCR data."""
import pandas as pd
import re
import collections

df = pd.read_csv(r"D:\My\University\AIC2026\AIC-26\data\ocr_results.csv")

for ch in ["ö", "ä", "ü", "�"]:
    words = df["text_full"].dropna().str.findall(rf"\S*{re.escape(ch)}\S*")
    flat = [w for sublist in words for w in sublist]
    c = collections.Counter(flat)
    print(f'=== Words with "{ch}" (top 15) ===')
    for w, cnt in c.most_common(15):
        print(f"  {w!r}: {cnt}")
    print()

# Also check H7T pattern
h7t = df["text_full"].dropna().str.findall(r"H7T|H7t")
flat_h7t = [w for sublist in h7t for w in sublist]
print(f"H7T count: {len(flat_h7t)}")
