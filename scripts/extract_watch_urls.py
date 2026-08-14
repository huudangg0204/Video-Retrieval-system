#!/usr/bin/env python3
"""
Script check từng file JSON trong folder media-info và trích xuất watch_url,
sau đó nhóm theo từng tập video (ví dụ L21, L22,...) và xuất ra các file JSON tương ứng (ví dụ L21.json, L22.json,...).

Sử dụng:
    python scripts/extract_watch_urls.py
    python scripts/extract_watch_urls.py --input-dir media-info --output-dir watch_urls
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


def extract_group_key(filename_stem: str) -> str:
    """
    Xác định prefix nhóm từ tên file.
    Ví dụ: 'L21_V001' -> 'L21', 'L30_V096' -> 'L30'.
    Nếu không khớp pattern prefix_Vxxx, lấy phần trước dấu gạch dưới đầu tiên.
    """
    match = re.match(r"^(.+?)_V\d+", filename_stem, re.IGNORECASE)
    if match:
        return match.group(1)
    if "_" in filename_stem:
        return filename_stem.split("_")[0]
    return filename_stem


def process_media_info(
    input_dir: Path,
    output_dir: Path,
    fmt: str = "dict"
) -> Dict[str, int]:
    """
    Đọc tất cả file json trong input_dir, lấy watch_url và lưu ra output_dir theo từng nhóm video.
    """
    if not input_dir.exists():
        print(f"❌ Lỗi: Thư mục đầu vào '{input_dir}' không tồn tại!", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"⚠️  Không tìm thấy file .json nào trong '{input_dir}'", file=sys.stderr)
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary gom nhóm dữ liệu: group_key -> list các cặp (video_id, watch_url)
    groups: Dict[str, List[tuple[str, str]]] = {}
    total_processed = 0
    missing_url_count = 0

    for file_path in json_files:
        video_id = file_path.stem  # Ví dụ: L21_V001
        group_key = extract_group_key(video_id)  # Ví dụ: L21

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            watch_url = data.get("watch_url", "")
            if not watch_url:
                missing_url_count += 1
                watch_url = data.get("url", "")

            if group_key not in groups:
                groups[group_key] = []

            groups[group_key].append((video_id, watch_url))
            total_processed += 1
        except Exception as e:
            print(f"⚠️  Lỗi khi đọc file {file_path.name}: {e}", file=sys.stderr)

    # Ghi ra các file json theo từng nhóm
    generated_files = {}
    for group_key, items in sorted(groups.items()):
        out_file = output_dir / f"{group_key}.json"

        if fmt == "dict":
            # Dạng dictionary: { "L21_V001": "https://youtube.com/watch?v=...", ... }
            content: Any = {video_id: url for video_id, url in items}
        elif fmt == "list":
            # Dạng danh sách các object: [ {"video_id": "L21_V001", "url": "https://..."}, ... ]
            content = [{"video_id": video_id, "url": url} for video_id, url in items]
        elif fmt == "pairs":
            # Dạng danh sách các cặp: [ ["L21_V001", "https://..."], ... ]
            content = [[video_id, url] for video_id, url in items]
        else:
            content = {video_id: url for video_id, url in items}

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

        generated_files[group_key] = len(items)

    print(f"✅ Đã xử lý thành công {total_processed} file video.")
    if missing_url_count > 0:
        print(f"⚠️  Có {missing_url_count} file không có 'watch_url'.")

    print(f"📁 Đã xuất {len(generated_files)} file JSON vào thư mục '{output_dir}':")
    for group_key, count in generated_files.items():
        print(f"   - {group_key}.json ({count} video)")

    return generated_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trích xuất watch_url từ các file media-info JSON và nhóm thành các file JSON theo từng video set."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=Path("media-info"),
        help="Thư mục chứa các file JSON media-info (Mặc định: media-info)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("watch_urls"),
        help="Thư mục đầu ra chứa các file JSON theo nhóm video (Mặc định: watch_urls)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["dict", "list", "pairs"],
        default="dict",
        help="Định dạng cấu trúc lưu trong file JSON output (dict: {video_id: url}, list: [{video_id, url}], pairs: [[video_id, url]])"
    )

    args = parser.parse_args()
    process_media_info(args.input_dir, args.output_dir, args.format)


if __name__ == "__main__":
    main()
