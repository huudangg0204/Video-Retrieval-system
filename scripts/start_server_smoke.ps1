# Start the AIC-26 API server against the lightweight *_smoke/ dataset
# instead of the full real dataset — keeps model/index load light enough
# for a 2GB-VRAM machine. Run this in ONE terminal and leave it running;
# run test_api_smoke.py in a SECOND terminal against it.
#
# Usage:
#   .\start_server_smoke.ps1

$env:DEVICE = "cpu"
$env:YOLO_MODEL_NAME = "yolo26n.pt"
$env:KEYFRAMES_DIR = "data/keyframes_smoke"
$env:EMBEDDING_DIR = "data/embeddings_smoke"
$env:CLIP32_EMBEDDING_DIR = "data/clip32_embeddings_smoke"
$env:DETECTION_DIR = "data/detections_smoke"
$env:METADATA_DIR = "data/metadata_smoke"
$env:DB_PATH = "data/faiss_db_smoke.pkl"
$env:FAISS_INDEX_PATH = "data/faiss_index_smoke.pkl"
$env:OCR_CSV_PATH = "data/ocr_results_smoke.csv"
$env:OCR_OUTPUT_PATH = "data/ocr_results_smoke.csv"

Write-Host "Starting API server against *_smoke/ dataset (device=cpu, yolo=nano)..." -ForegroundColor Cyan
Write-Host "Once you see 'Models loaded. Server is now ready...', open a second terminal and run test_api_smoke.py" -ForegroundColor Yellow

aic26 serve api