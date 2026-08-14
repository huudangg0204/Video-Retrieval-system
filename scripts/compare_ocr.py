"""Compare before/after OCR post-processing."""
import pandas as pd, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

df_b = pd.read_csv(r"D:\My\University\AIC2026\AIC-26\data\ocr_results.csv")
df_a = pd.read_csv(r"D:\My\University\AIC2026\AIC-26\data\ocr_results_clean.csv")

patterns = [
    ("Han Qu", "Vietnamese ö fix"),
    ("2O2", "Digit O→0 fix"),
    ("1853:", "Time format fix"),
    ("d@", "@ → ê fix"),
    ("Bäc", "German ä fix"),
]

for pat, desc in patterns:
    mask = df_b["text_full"].str.contains(re.escape(pat), na=False)
    idxs = mask[mask].index[:2]
    if len(idxs) == 0:
        continue
    print(f"=== {desc} (pattern: '{pat}') ===")
    for i in idxs:
        b = str(df_b.loc[i, "text_full"])[:120]
        a = str(df_a.loc[i, "text_full"])[:120]
        print(f"  BEFORE: {b}")
        print(f"  AFTER:  {a}")
        print()
