import os
import pandas as pd
import csv

GAP_KEYFRAME = 15 # Video thường 25 -> 30fps, để 15 cho chắc 

def make_submission(csv_file_name, video_name, frame_index, const_value=None):
    rows = []
    rows.append((video_name, frame_index, const_value) if const_value is not None else (video_name, frame_index))

    # Tạo 19 dòng tiếp theo
    for i in range(1, 10):  # mỗi i tạo ra 2 dòng (+ và -)
        plus_idx = frame_index + i * GAP_KEYFRAME
        minus_idx = frame_index - i * GAP_KEYFRAME

        if const_value is not None:
            rows.append((video_name, plus_idx, const_value))
            rows.append((video_name, minus_idx, const_value))
        else:
            rows.append((video_name, plus_idx))
            rows.append((video_name, minus_idx))

    # Chỉ lấy đúng 20 dòng
    rows = rows[:20]

    # Ghi ra CSV
    with open(f"{csv_file_name}.csv", mode="w", encoding='utf-8',newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

# Ví dụ chạy thử
make_submission("query-p2-19-kis", "L22_V011", 13415)