import sys
from pathlib import Path
from aic26.config.settings import Settings
from aic26.pipeline.orchestrator import PipelineOrchestrator

def main():
    settings = Settings()
    # Ensure all directories point to the correct data folder
    settings.keyframes_dir = Path("data/keyframes")
    
    print(f"keyframes_dir = {settings.keyframes_dir}")
    print("Running full pipeline on remaining folders (skip existing)")

    orchestrator = PipelineOrchestrator(config=settings)
    results = orchestrator.run(
        video_dir="data/videos_temp",
        skip_detection=False,
        skip_ocr=False,
        skip_embeddings=False,
        skip_index=False,
        skip_metadata=False,
        skip_scene=False,
        skip_scene_graph=True,
        skip_captions=False,
    )
    print("Done!", results.get("_pipeline_stats"))

if __name__ == "__main__":
    main()
