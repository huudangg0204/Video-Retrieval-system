"""
Keyframe extraction from video files.

Uses DINOv2 embeddings and cosine similarity to detect 
scene changes and extract representative keyframes.

Migrated from: create.py
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

from loguru import logger

from aic26.config import Settings
from aic26.utils.file_utils import ensure_dir


class KeyframeExtractor:
    """
    Extract keyframes from videos based on visual differences.
    
    Uses DINOv2 for frame embeddings and cosine similarity
    to detect scene changes. Frames are sampled every N seconds
    and kept if they differ significantly from the last keyframe.
    
    Usage:
        extractor = KeyframeExtractor(settings)
        extractor.process("./videos", "./keyframes")
    """
    
    def __init__(self, config: Optional['Settings'] = None) -> None:
        """
        Initialize the keyframe extractor.
        
        Args:
            config: Settings instance (uses global settings if None)
        """
        if config is None:
            from aic26.config import settings as global_settings
            config = global_settings
        
        self.config = config
        requested = (getattr(config, "device", "cuda") or "cuda").lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(f"[DEVICE] config.device={requested!r} but CUDA unavailable — using cpu")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(requested)
        
        logger.info(f"Loading DINOv2 model on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.model = AutoModel.from_pretrained('facebook/dinov2-small').to(self.device).eval()
        logger.info("DINOv2 model ready.")
    
    def process(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        num_workers: int = 2,
    ) -> dict:
        """
        Process all videos in a directory.
        
        Args:
            input_dir: Directory containing video files
            output_dir: Output directory (uses config default if None)
            num_workers: Number of parallel workers
            
        Returns:
            Dict with processing statistics
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path = Path(output_dir) if output_dir else Path(self.config.keyframes_dir)
        map_path = Path(self.config.map_keyframe_dir)
        
        ensure_dir(output_path)
        ensure_dir(map_path)
        
        # Find video files
        video_files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
        ]
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return {"videos_found": 0, "videos_processed": 0, "keyframes_total": 0}
        
        logger.info(f"Found {len(video_files)} video(s) to process")
        
        stats = {"videos_found": len(video_files), "videos_processed": 0, "keyframes_total": 0}
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._extract_from_video,
                    str(input_path / video_file),
                    str(output_path),
                    str(map_path),
                ): video_file
                for video_file in video_files
            }
            
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing videos",
                unit="video",
            ):
                video_file = futures[future]
                try:
                    result = future.result()
                    if result:
                        stats["videos_processed"] += 1
                        stats["keyframes_total"] += result.get("keyframes", 0)
                except Exception as e:
                    logger.error(f"Failed to process {video_file}: {e}")
        
        logger.info(
            f"Keyframe extraction complete: "
            f"{stats['videos_processed']}/{stats['videos_found']} videos, "
            f"{stats['keyframes_total']} keyframes total"
        )
        
        return stats
    
    def _extract_from_video(
        self,
        video_path: str,
        output_dir: str,
        map_dir: str,
    ) -> Optional[dict]:
        """
        Extract keyframes from a single video.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory for keyframe images
            map_dir: Directory for mapping CSV files
            
        Returns:
            Dict with extraction stats, or None if skipped
        """
        video_name = Path(video_path).stem
        video_output = os.path.join(output_dir, video_name)
        map_file = os.path.join(map_dir, f"{video_name}.csv")
        
        # Skip if already processed
        if os.path.exists(map_file):
            logger.debug(f"Already processed: {video_name}")
            return {"video_name": video_name, "keyframes": 0, "skipped": True}
        
        ensure_dir(video_output)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.debug(f"{video_name}: {total_frames} frames @ {fps:.1f} fps")
        
        map_data = []
        last_frame_vector = None
        last_frame_time = None
        saved_count = 1
        current_frame_idx = 0
        
        pbar = tqdm(
            total=total_frames,
            desc=f"  {video_name}",
            unit="frames",
            leave=False,
        )
        
        while current_frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                frames_to_skip = int(fps)
                current_frame_idx += frames_to_skip
                pbar.update(frames_to_skip)
                continue
            
            pst_time = current_frame_idx / fps
            current_vector = self._get_image_vector(frame)
            
            # Accept frame if:
            # 1. First frame
            # 2. Time since last > auto_accept_seconds
            # 3. Cosine similarity < threshold
            if last_frame_vector is None:
                is_accepted = True
            elif pst_time - last_frame_time >= self.config.keyframe_auto_accept_seconds:
                is_accepted = True
            else:
                similarity = self._calculate_cosine(last_frame_vector, current_vector)
                is_accepted = similarity < self.config.keyframe_similarity_threshold
            
            if is_accepted:
                frame_filename = os.path.join(video_output, f"{saved_count:03d}.jpg")
                cv2.imwrite(frame_filename, frame)
                
                map_data.append({
                    "n": saved_count,
                    "pst_time": pst_time,
                    "fps": fps,
                    "frame_index": current_frame_idx,
                })
                
                last_frame_vector = current_vector
                last_frame_time = pst_time
                saved_count += 1
            
            # Skip to next check point
            frames_to_skip = max(1, int(self.config.keyframe_interval * fps))
            current_frame_idx += frames_to_skip
            pbar.update(frames_to_skip)
        
        pbar.close()
        cap.release()
        
        # Save mapping CSV
        if saved_count > 1:
            df = pd.DataFrame(map_data)
            df.to_csv(map_file, index=False)
            logger.debug(f"{video_name}: {saved_count - 1} keyframes extracted")
        
        return {
            "video_name": video_name,
            "keyframes": saved_count - 1,
            "skipped": False,
        }
    
    def _get_image_vector(self, image: np.ndarray) -> np.ndarray:
        """
        Extract DINOv2 embedding from an image.
        
        Args:
            image: BGR image as numpy array (from cv2)
            
        Returns:
            Embedding vector as numpy array
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        if hasattr(outputs, "pooler_output"):
            vec = outputs.pooler_output[0]
        else:
            vec = outputs.last_hidden_state.mean(dim=1)[0]
        
        return vec.detach().cpu().numpy()
    
    @staticmethod
    def _calculate_cosine(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))