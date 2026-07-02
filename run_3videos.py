"""Run pipeline for only 3 videos using keyframes_3 junction dir.
Skip detection (already done) to free VRAM for embeddings.
"""
from pathlib import Path
from aic26.config.settings import Settings
from aic26.pipeline.orchestrator import PipelineOrchestrator

# Override keyframes_dir to point to the 3-video junction
settings = Settings()
settings.keyframes_dir = Path("data/keyframes_3")

print(f"keyframes_dir = {settings.keyframes_dir}")
print(f"ocr_csv_path  = {settings.ocr_csv_path}")

orchestrator = PipelineOrchestrator(config=settings)
results = orchestrator.run(
    video_dir="data/videos_temp",
    skip_detection=True,     # already done
    skip_ocr=True,           # use pre-computed CSV
    skip_scene_graph=True,   # too heavy for RTX 3050
)
print("Done!", results.get("_pipeline_stats"))
