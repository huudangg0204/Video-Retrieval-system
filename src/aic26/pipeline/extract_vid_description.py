"""
Video Description Extraction Pipeline using Gemini API
======================================================
Trích xuất mô tả phân đoạn video YouTube qua Gemini API.
Hỗ trợ Save Progress/Resume, rate limiting, thông báo Video ID bị lỗi.

Yêu cầu cài đặt SDK:
    pip install google-genai
HOẶC
    pip install google-generativeai

Sử dụng:
    export GEMINI_API_KEY="your_api_key"
    python src/aic26/pipeline/extract_vid_description.py \
        --input watch_urls/L29.json \
        --output data/vid-description/L29.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Load biến môi trường từ file .env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Thử import tqdm (nếu không có thì dùng dummy iterator)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Import Gemini SDK (ưu tiên SDK mới google-genai, nếu không có thì fallback sang google-generativeai)
GENAI_SDK_TYPE = None
try:
    from google import genai
    from google.genai import types
    GENAI_SDK_TYPE = "google-genai"
except ImportError:
    try:
        import google.generativeai as legacy_genai
        GENAI_SDK_TYPE = "google-generativeai"
    except ImportError:
        GENAI_SDK_TYPE = None

# Config mặc định
DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_DELAY = 4.0
RETRY_ATTEMPTS = 3

PROMPT = """Xem video YouTube này và mô tả toàn bộ những gì xảy ra từ đầu đến cuối.
KHÔNG bỏ qua bất kỳ khoảnh khắc nào dù không liên quan đến chủ đề chính.

Trả về JSON duy nhất với cấu trúc sau:

{
  "segments": [
    {
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS",
      "description": "Cố gắng mô tả cụ thể những gì đang xảy ra, không bỏ sót khoảnh khắc nào",
      "category": "thể loại/danh mục của phân đoạn",
      "entities": ["danh sách các đối tượng, tên người, tổ chức, địa danh xuất hiện trong đoạn"],
      "keywords": ["3-5 từ khoá đại diện cho khoảnh khắc này"]
    }
  ]
}

Yêu cầu bắt buộc:
- mỗi segment có độ dài không quá 4 phút, nếu không có hành động, sự kiện cụ thể thì có thể hơn.
- Segments liên tục, không có khoảng trống thời gian giữa các đoạn
- Bắt đầu từ 00:00:00, kết thúc tại thời điểm cuối video
- Định dạng thời gian start_time và end_time bắt buộc dạng HH:MM:SS (ví dụ 00:01:15)

Chỉ trả về JSON hợp lệ, không thêm văn bản giải thích nào khác."""

logger = logging.getLogger("extract_vid_description")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def call_gemini_google_genai(client, model_name: str, video_url: str) -> dict:
    """Gọi Gemini qua SDK mới google-genai với YouTube URL."""
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_uri(
                file_uri=video_url,
                mime_type="video/mp4",
            ),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


def call_gemini_legacy(model, video_url: str) -> dict:
    """Gọi Gemini qua SDK legacy google-generativeai với YouTube URL."""
    response = model.generate_content(
        contents=[
            {
                "parts": [
                    {"text": PROMPT},
                    {
                        "file_data": {
                            "mime_type": "video/mp4",
                            "file_uri": video_url,
                        }
                    },
                ]
            }
        ],
        generation_config=legacy_genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


def save_json_atomically(data: dict, file_path: Path):
    """Lưu file JSON an toàn (Save Progress)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(file_path)


def process_single_video(client_or_model, sdk_type: str, model_name: str, video_id: str, video_url: str) -> dict:
    """Xử lý 1 video với cơ chế auto-retry exponential backoff tự viết."""
    last_exception = None
    wait_time = 4.0

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            if sdk_type == "google-genai":
                result = call_gemini_google_genai(client_or_model, model_name, video_url)
            else:
                result = call_gemini_legacy(client_or_model, video_url)

            segments = result.get("segments", [])
            return {"segments": segments}

        except Exception as e:
            last_exception = e
            logger.warning(f"Lần thử {attempt}/{RETRY_ATTEMPTS} cho video {video_id} thất bại: {e}. Thử lại sau {wait_time:.1f}s...")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(wait_time)
                wait_time *= 2.0

    raise last_exception


def run_pipeline(
    input_file: Path,
    output_file: Path,
    model_name: str = DEFAULT_MODEL,
    delay_seconds: float = DEFAULT_DELAY,
):
    setup_logging()

    if not GENAI_SDK_TYPE:
        logger.error("Không tìm thấy Gemini SDK. Vui lòng cài đặt: pip install google-genai HOẶC pip install google-generativeai")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("Thiếu biến môi trường GEMINI_API_KEY. Vui lòng chạy: export GEMINI_API_KEY='your_api_key'")
        sys.exit(1)

    # Khởi tạo Client/Model theo SDK khả dụng
    if GENAI_SDK_TYPE == "google-genai":
        logger.info(f"Sử dụng Gemini SDK mới (google-genai) với model: {model_name}")
        client_or_model = genai.Client(api_key=api_key)
    else:
        logger.info(f"Sử dụng Gemini SDK legacy (google-generativeai) với model: {model_name}")
        legacy_genai.configure(api_key=api_key)
        client_or_model = legacy_genai.GenerativeModel(model_name)

    # 1. Load Input JSON
    if not input_file.exists():
        logger.error(f"File đầu vào không tồn tại: {input_file}")
        sys.exit(1)

    with open(input_file, encoding="utf-8") as f:
        urls_data: dict = json.load(f)

    logger.info(f"Tổng số video trong input ({input_file.name}): {len(urls_data)}")

    # 2. Load Output JSON nếu đã có (Save Progress / Resume)
    output_data = {}
    if output_file.exists():
        try:
            with open(output_file, encoding="utf-8") as f:
                output_data = json.load(f)
            logger.info(f"Đã nạp file output cũ ({output_file.name}): {len(output_data)} video đã có.")
        except Exception as e:
            logger.warning(f"Không thể đọc file output cũ ({e}), sẽ tạo mới.")

    # 3. Lọc danh sách video cần xử lý
    todo_videos = {vid_id: url for vid_id, url in urls_data.items() if vid_id not in output_data}

    if not todo_videos:
        logger.info("Tất cả video trong file input đã hoàn thành! Bỏ qua processing.")
        return

    logger.info(f"Cần xử lý: {len(todo_videos)} video chưa hoàn thành (Bỏ qua: {len(urls_data) - len(todo_videos)} video đã hoàn thành).")

    failed_videos = {}

    # 4. Tiến hành xử lý từng video
    pbar = tqdm(todo_videos.items(), desc="Extracting Descriptions", unit="video")
    for video_id, url in pbar:
        try:
            video_result = process_single_video(
                client_or_model,
                GENAI_SDK_TYPE,
                model_name,
                video_id,
                url,
            )

            # Cập nhật kết quả vào memory & Save Progress lập tức
            output_data[video_id] = video_result
            save_json_atomically(output_data, output_file)
            logger.info(f"✓ Hoàn thành {video_id} -> Đã lưu vào {output_file.name}")

        except Exception as e:
            error_msg = str(e)
            failed_videos[video_id] = error_msg
            logger.error(f"✗ [LỖI / RATE LIMIT] Video ID '{video_id}' thất bại! Nguyên nhân: {error_msg}")

        # Delay giữa các call để phòng Rate Limit (Free Tier)
        time.sleep(delay_seconds)

    # 5. Tổng kết
    logger.info("\n" + "=" * 50)
    logger.info(f"HOÀN THÀNH PIPELINE:")
    logger.info(f"  - Tổng số video thành công trong file output: {len(output_data)} / {len(urls_data)}")

    if failed_videos:
        logger.error(f"  - Số video BỊ LỖI / RATE LIMIT: {len(failed_videos)}")
        logger.error("Danh sách các Video ID bị lỗi:")
        for vid_id, err in failed_videos.items():
            logger.error(f"    + Video ID: {vid_id} | Lỗi: {err}")
        logger.info("Bạn có thể chạy lại script để resume các video bị lỗi sau khi hết cooldown Rate Limit.")
    else:
        logger.info("  - Tất cả video đều đã trích xuất thành công 100%!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Video Descriptions using Gemini API")
    parser.add_argument("--input", "-i", required=True, help="Đường dẫn file JSON input (vd: watch_urls/L29.json)")
    parser.add_argument("--output", "-o", required=True, help="Đường dẫn file JSON output (vd: data/vid-description/L29.json)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Gemini Model Name (Default: {DEFAULT_MODEL})")
    parser.add_argument("--delay", "-d", type=float, default=DEFAULT_DELAY, help=f"Giây delay giữa các request (Default: {DEFAULT_DELAY})")

    args = parser.parse_args()

    run_pipeline(
        input_file=Path(args.input),
        output_file=Path(args.output),
        model_name=args.model,
        delay_seconds=args.delay,
    )