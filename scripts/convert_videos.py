#!/usr/bin/env python3
"""
Utility script to batch check and convert video formats for the AIC-26 pipeline.
"""
import sys
from pathlib import Path
from loguru import logger

# Add src to PYTHONPATH so we can import aic26
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aic26.utils.video_converter import convert_video_for_pipeline

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check and convert videos to pipeline-compatible format (MP4 H.264)")
    parser.add_argument("input_dir", help="Directory containing videos to process")
    parser.add_argument("--output_dir", "-o", help="Optional output directory for converted videos", default=None)
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Directory not found: {input_dir}")
        sys.exit(1)
        
    videos = [
        f for f in input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.av1', '.ts']
    ]
    
    logger.info(f"Found {len(videos)} video candidates to check...")
    
    success_count = 0
    for v in videos:
        # Skip files that were already converted by this script (if saving in same dir)
        if v.name.endswith('_converted.mp4'):
            continue
            
        try:
            convert_video_for_pipeline(v, output_dir=args.output_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing {v.name}: {e}")

    logger.info(f"Finished checking/converting. Successful: {success_count}/{len(videos)}")

if __name__ == "__main__":
    main()
