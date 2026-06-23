import os
import subprocess
from pathlib import Path
from loguru import logger

def is_video_supported(video_path: str | Path) -> bool:
    """
    Check if the video is natively supported by the pipeline.
    We check if it's an MP4 file with H.264 or HEVC codec.
    """
    path = str(video_path)
    if not path.lower().endswith(".mp4"):
        return False
        
    try:
        # Use ffprobe to get video codec
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream=codec_name", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        codec = result.stdout.strip()
        
        # H.264 and HEVC are the most widely supported formats in cv2
        return codec.lower() in ("h264", "hevc")
    except FileNotFoundError:
        logger.warning("ffprobe not found. Cannot check video codec. Assuming unsupported to be safe.")
        return False
    except Exception as e:
        logger.warning(f"Could not check codec for {path}: {e}")
        return False

def convert_video_for_pipeline(input_path: str | Path, output_dir: str | Path = None) -> Path:
    """
    Convert a video to an MP4 (H.264) format suitable for the pipeline.
    
    Args:
        input_path: Path to the original video
        output_dir: Optional directory to save the converted video. 
                    If None, saves in the same directory with a _converted suffix.
                    
    Returns:
        Path to the converted video.
    """
    in_path = Path(input_path)
    
    if not in_path.exists():
        raise FileNotFoundError(f"Video not found: {in_path}")
        
    if is_video_supported(in_path):
        logger.info(f"Video '{in_path.name}' is already in a supported format (MP4 / H.264). No conversion needed.")
        return in_path
        
    # Determine output path
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{in_path.stem}.mp4"
    else:
        out_path = in_path.parent / f"{in_path.stem}_converted.mp4"
        
    # Skip if converted file already exists
    if out_path.exists() and out_path != in_path:
        logger.info(f"Converted video already exists at: {out_path}")
        return out_path
        
    logger.info(f"Converting video '{in_path.name}' to standard MP4 (H.264)... This may take a while.")
    
    try:
        cmd = [
            "ffmpeg",
            "-y",                   # Overwrite output
            "-i", str(in_path),     # Input file
            "-c:v", "libx264",      # Video codec: H.264
            "-preset", "fast",      # Encoding speed/compression tradeoff
            "-crf", "23",           # Constant Rate Factor (visual quality)
            "-c:a", "aac",          # Audio codec: AAC
            "-b:a", "128k",         # Audio bitrate
            str(out_path)           # Output file
        ]
        
        # Run ffmpeg silently
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        logger.info(f"✅ Successfully converted video: {out_path}")
        return out_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed for {in_path.name}: {e}")
        # Clean up partial file if conversion failed
        if out_path.exists() and out_path != in_path:
            out_path.unlink()
        raise RuntimeError(f"Failed to convert video {in_path.name}") from e
    except FileNotFoundError:
        logger.error("ffmpeg is not installed on the system. Please install ffmpeg to enable video conversion.")
        raise
