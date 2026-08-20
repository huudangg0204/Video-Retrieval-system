#!/usr/bin/env python3
"""
Script chuẩn hóa và tối ưu hóa file ocr_results.csv cho AIC-26.

Thực hiện 2 nhiệm vụ chính:
1. Format thống nhất đường dẫn keyframe: {base_dir}/{video}/{file}
   (loại bỏ các đường dẫn Kaggle / Windows / path không đồng nhất).
2. Loại bỏ các cột thừa (boxes, texts, confs) để giảm dung lượng file
   và tối ưu RAM / tốc độ tra cứu cho OCR Search Engine.

Usage:
    python scripts/clean_ocr_csv.py
    python scripts/clean_ocr_csv.py --input data/ocr_results.csv --output data/ocr_results.csv --backup
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_CSV = "data/ocr_results.csv"
DEFAULT_BASE_DIR = "/home/huudang/Documents/projects/AIC-26/data/keyframes"
ESSENTIAL_COLUMNS = ["video", "file", "path", "text_full", "num_boxes", "conf_mean"]


def clean_ocr_csv(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    base_dir: str = DEFAULT_BASE_DIR,
    backup: bool = True,
) -> dict:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {input_path}")

    output_path = Path(output_csv) if output_csv else input_path
    base_dir = base_dir.rstrip("/\\")

    # Kích thước ban đầu
    size_before_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"[*] Đang nạp {input_path} ({size_before_mb:.2f} MB)...")

    # Đọc CSV
    df = pd.read_csv(input_path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    total_rows = len(df)
    print(f"[*] Đã tải {total_rows:,} dòng.")
    print(f"[*] Các cột hiện có: {df.columns.tolist()}")

    # 1. Chuẩn hóa đường dẫn path
    print(f"[*] Chuẩn hóa đường dẫn keyframe theo định dạng: '{base_dir}/<video>/<file>'...")
    clean_video = df["video"].astype(str).str.strip()
    clean_file = df["file"].astype(str).str.strip()
    df["video"] = clean_video
    df["file"] = clean_file
    df["path"] = base_dir + "/" + clean_video + "/" + clean_file

    # 2. Xử lý các cột nội dung & số liệu
    if "text_full" in df.columns:
        df["text_full"] = df["text_full"].fillna("").astype(str).str.strip()
    else:
        df["text_full"] = ""

    if "num_boxes" in df.columns:
        df["num_boxes"] = pd.to_numeric(df["num_boxes"], errors="coerce").fillna(0).astype(int)
    else:
        df["num_boxes"] = 0

    if "conf_mean" in df.columns:
        df["conf_mean"] = pd.to_numeric(df["conf_mean"], errors="coerce").fillna(0.0).round(4)
    else:
        df["conf_mean"] = 0.0

    # 3. Lọc bỏ các cột thừa, chỉ giữ lại ESSENTIAL_COLUMNS
    dropped_cols = [c for c in df.columns if c not in ESSENTIAL_COLUMNS]
    df_clean = df[ESSENTIAL_COLUMNS].copy()
    print(f"[*] Đã loại bỏ các cột thừa: {dropped_cols}")
    print(f"[*] Các cột giữ lại: {df_clean.columns.tolist()}")

    # 4. Loại bỏ dòng trùng lặp (nếu có)
    dups = df_clean.duplicated(subset=["video", "file"]).sum()
    if dups > 0:
        print(f"[*] Loại bỏ {dups:,} dòng trùng lặp (video, file)...")
        df_clean = df_clean.drop_duplicates(subset=["video", "file"]).reset_index(drop=True)

    # 5. Tạo backup nếu ghi đè
    if backup and input_path.resolve() == output_path.resolve():
        bak_path = input_path.with_suffix(input_path.suffix + ".bak")
        print(f"[*] Tạo bản sao lưu backup tại: {bak_path}")
        shutil.copy2(input_path, bak_path)

    # 6. Ghi ra file output
    print(f"[*] Đang lưu file kết quả ra: {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False, encoding="utf-8")

    size_after_mb = output_path.stat().st_size / (1024 * 1024)
    saved_mb = size_before_mb - size_after_mb
    percent_saved = (saved_mb / max(size_before_mb, 0.001)) * 100

    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT CHUẨN HÓA OCR CSV!")
    print("=" * 60)
    print(f"  - Số dòng sau xử lý:    {len(df_clean):,}")
    print(f"  - Kích thước ban đầu:  {size_before_mb:.2f} MB")
    print(f"  - Kích thước sau clean: {size_after_mb:.2f} MB")
    print(f"  - Dung lượng tiết kiệm: {saved_mb:.2f} MB ({percent_saved:.1f}%)")
    print(f"  - Output:               {output_path}")
    print("=" * 60)

    print("\nMẫu 3 dòng đầu tiên:")
    for idx, row in df_clean.head(3).iterrows():
        print(f"  [{row['video']}/{row['file']}] -> {row['path']}")
        print(f"   Text: {row['text_full'][:80]} | Boxes: {row['num_boxes']} | Conf: {row['conf_mean']}")

    return {
        "rows": len(df_clean),
        "size_before_mb": size_before_mb,
        "size_after_mb": size_after_mb,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Clean and standardize OCR results CSV.")
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT_CSV,
        help=f"Đường dẫn file OCR CSV đầu vào (mặc định: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Đường dẫn file đầu ra (mặc định: ghi đè file input)",
    )
    parser.add_argument(
        "--base-dir", "-b",
        default=DEFAULT_BASE_DIR,
        help=f"Đường dẫn gốc keyframes (mặc định: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Không tạo file backup .bak khi ghi đè",
    )

    args = parser.parse_args()
    clean_ocr_csv(
        input_csv=args.input,
        output_csv=args.output,
        base_dir=args.base_dir,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
