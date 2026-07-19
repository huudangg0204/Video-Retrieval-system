"""Test API returns clean OCR text."""
import requests, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

r = requests.post("http://localhost:5000/api/search_text", json={"query": "news", "top_k": 3})
data = r.json()
print(f"Status: {r.status_code}, Count: {data.get('count')}\n")
for res in data.get("results", []):
    vid = res["video_name"]
    frame = res["frame_idx"]
    ocr = res.get("ocr_text", [])
    texts = [o["text"] for o in ocr if isinstance(o, dict)] if isinstance(ocr, list) else [str(ocr)]
    print(f"{vid}/{frame}")
    print(f"  OCR: {' | '.join(texts)}")
    print()
