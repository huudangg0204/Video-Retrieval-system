"""
Object detection on keyframes using YOLOv8.

Processes keyframe images to detect objects and saves
detection results as JSON files with MID/Entity mappings.

Migrated from: obj_detect.py, batch_process_obj_kf.py
"""
from __future__ import annotations

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm

from loguru import logger

from aic26.config import Settings
from aic26.utils.file_utils import ensure_dir


class ObjectDetector:
    """
    Detect objects in keyframe images using YOLOv8.
    
    Supports Open Images V7 class mapping and batch processing
    with multi-threading support.
    
    Usage:
        detector = ObjectDetector(settings)
        detector.process("./keyframes", "./detections")
    """
    
    def __init__(self, config: Optional['Settings'] = None) -> None:
        """
        Initialize the object detector.
        
        Args:
            config: Settings instance
        """
        if config is None:
            from aic26.config import settings as global_settings
            config = global_settings
        
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load mappings
        self.mid_to_entity, self.entity_to_mid = self._load_mapping()
        
        # Load model (lazy)
        self._model = None
    
    def _load_model(self):
        """Lazy-load YOLO model."""
        if self._model is None:
            from ultralytics import YOLO
            
            logger.info(f"Loading YOLOv8 model on {self.device}...")
            model_path = self.config.yolo_model_name
            
            self._model = YOLO(model_path)
            self._model.overrides['verbose'] = False
            logger.info("YOLOv8 model ready.")
    
    def _load_mapping(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Load Open Images V7 MID ↔ Entity name mappings.
        
        Returns:
            Tuple of (mid_to_entity, entity_to_mid) dictionaries
        """
        mid_to_entity = {}
        entity_to_mid = {}
        
        mapping_path = self.config.oid_mapping_csv
        if not os.path.exists(mapping_path):
            logger.warning(f"Mapping file not found: {mapping_path}")
            return mid_to_entity, entity_to_mid
        
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                for mid, entity in csv.reader(f):
                    mid_to_entity[mid] = entity
                    entity_to_mid[entity.lower()] = mid
            logger.debug(f"Loaded {len(mid_to_entity)} class mappings")
        except Exception as e:
            logger.error(f"Failed to load mapping: {e}")
        
        return mid_to_entity, entity_to_mid
    
    def process(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        num_workers: int = 4,
    ) -> dict:
        """
        Process all video folders in the keyframes directory.
        
        Args:
            input_dir: Directory containing video subfolders with keyframes
            output_dir: Output directory for detection JSONs
            num_workers: Number of parallel workers
            
        Returns:
            Dict with processing statistics
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path = Path(output_dir) if output_dir else Path(self.config.detection_dir)
        ensure_dir(output_path)
        
        # Load model
        self._load_model()
        
        # Find video folders
        video_folders = sorted(
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        )
        
        if not video_folders:
            logger.warning(f"No video folders found in {input_dir}")
            return {"folders_found": 0, "folders_processed": 0, "images_total": 0}
        
        logger.info(f"Found {len(video_folders)} video folder(s) to process")
        
        stats = {
            "folders_found": len(video_folders),
            "folders_processed": 0,
            "images_total": 0,
        }
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._process_folder,
                    str(input_path / folder),
                    str(output_path / folder),
                ): folder
                for folder in video_folders
            }
            
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Detecting objects",
                unit="folder",
            ):
                folder = futures[future]
                try:
                    result = future.result()
                    if result:
                        stats["folders_processed"] += 1
                        stats["images_total"] += result.get("images", 0)
                except Exception as e:
                    logger.error(f"Failed to process {folder}: {e}")
        
        logger.info(
            f"Object detection complete: "
            f"{stats['folders_processed']}/{stats['folders_found']} folders, "
            f"{stats['images_total']} images"
        )
        
        return stats
    
    def _process_folder(
        self,
        input_folder: str,
        output_folder: str,
    ) -> Optional[dict]:
        """
        Process all images in a single video folder.
        
        Args:
            input_folder: Folder containing keyframe images
            output_folder: Output folder for JSON results
            
        Returns:
            Dict with processing stats
        """
        ensure_dir(output_folder)
        
        # Find image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = sorted(
            f for f in os.listdir(input_folder)
            if f.lower().endswith(image_extensions)
        )
        
        if not image_files:
            logger.warning(f"No images in {input_folder}")
            return None
        
        processed = 0
        
        for image_file in tqdm(
            image_files,
            desc=f"  {os.path.basename(input_folder)}",
            unit="img",
            leave=False,
        ):
            image_path = os.path.join(input_folder, image_file)
            json_name = f"{Path(image_file).stem}.json"
            json_path = os.path.join(output_folder, json_name)
            
            try:
                detection = self._detect_single(image_path)
                
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(detection, f, indent=2, ensure_ascii=False)
                
                processed += 1
                
            except Exception as e:
                logger.warning(f"Error processing {image_file}: {e}")
        
        return {
            "folder": os.path.basename(input_folder),
            "images": processed,
        }
    
    def _detect_single(self, image_path: str) -> dict:
        """
        Detect objects in a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with detection results:
            - detection_scores: List of confidence scores
            - detection_class_names: List of MID identifiers
            - detection_class_entities: List of entity names
            - detection_boxes: List of [x1, y1, x2, y2] boxes
            - detection_class_labels: List of class indices
        """
        results = self._model.predict(
            source=image_path,
            conf=0.1,
            max_det=300,
            iou=0.7,
            nms_time_limit=10.0,
            device=self.device,
        )
        
        detection_scores = []
        detection_class_names = []
        detection_class_entities = []
        detection_boxes = []
        detection_class_labels = []
        
        for r in results:
            for box in r.boxes:
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                entity_name = self._model.names[cls_id]
                
                # Map to MID
                mid = self.entity_to_mid.get(
                    entity_name.strip().lower(),
                    "UNKNOWN",
                )
                
                detection_scores.append(score)
                detection_class_labels.append(cls_id)
                detection_class_names.append(mid)
                detection_class_entities.append(entity_name)
                detection_boxes.append([float(x) for x in box.xyxy[0]])
        
        return {
            "detection_scores": detection_scores,
            "detection_class_names": detection_class_names,
            "detection_class_entities": detection_class_entities,
            "detection_boxes": detection_boxes,
            "detection_class_labels": detection_class_labels,
        }