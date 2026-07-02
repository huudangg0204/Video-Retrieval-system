"""
Object detection on keyframes using YOLOv26.

Processes keyframe images to detect objects and saves
detection results as JSON files with MID/Entity mappings.

Migrated from: obj_detect.py, batch_process_obj_kf.py
"""

from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from aic26.config import Settings
from aic26.config.constants import POSITION_GRID_THRESHOLD
from aic26.pipeline.position import position_from_bbox
from aic26.utils.file_utils import ensure_dir


class ObjectDetector:
    """
    Detect objects in keyframe images using YOLOv26.

    Supports Open Images V7 class mapping and batch processing
    with multi-threading support.

    Usage:
        detector = ObjectDetector(settings)
        detector.process("./keyframes", "./detections")
    """

    def __init__(self, config: Settings | None = None) -> None:
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

            logger.info(f"Loading YOLOv26 model on {self.device}...")
            model_path = self.config.yolo_model_name

            self._model = YOLO(model_path)
            self._model.overrides["verbose"] = False
            logger.info("YOLOv26 model ready.")

    def _load_mapping(self) -> tuple[dict[str, str], dict[str, str]]:
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
            with open(mapping_path, encoding="utf-8") as f:
                # First line is the header "LabelName,DisplayName" — skip it.
                next(f, None)
                for mid, entity in csv.reader(f):
                    if not mid or not entity:
                        continue
                    mid_to_entity[mid] = entity
                    entity_to_mid[entity.lower()] = mid
            logger.debug(f"Loaded {len(mid_to_entity)} class mappings")
        except Exception as e:
            logger.error(f"Failed to load mapping: {e}")

        return mid_to_entity, entity_to_mid

    def process(
        self,
        input_dir: str,
        output_dir: str | None = None,
        num_workers: int = 1,
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
            d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
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
    ) -> dict | None:
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
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = sorted(
            f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)
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
        # Lazy-load YOLO weights on first call so callers don't need to
        # remember to call _load_model() explicitly.
        self._load_model()
        results = self._model.predict(
            source=image_path,
            conf=0.1,
            max_det=300,
            iou=0.7,
            device=self.device,
        )

        detection_scores = []
        detection_class_names = []
        detection_class_entities = []
        detection_boxes = []
        detection_class_labels = []
        # Image dimensions from the first result (all results share the same shape).
        # orig_shape is (height, width) in numpy convention.
        img_h: int | None = None
        img_w: int | None = None
        if results and len(results) > 0 and getattr(results[0], "orig_shape", None) is not None:
            img_h, img_w = int(results[0].orig_shape[0]), int(results[0].orig_shape[1])

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

        result = {
            "detection_scores": detection_scores,
            "detection_class_names": detection_class_names,
            "detection_class_entities": detection_class_entities,
            "detection_boxes": detection_boxes,
            "detection_class_labels": detection_class_labels,
        }
        # Enrich with per-detection grid positions when image dims are known.
        # Use the user-tunable threshold from Settings (.env override) so that
        # lowering POSITION_GRID_THRESHOLD for a domain with smaller objects
        # (e.g. text/OCR fragments) takes effect end-to-end without code changes.
        if img_h is not None and img_w is not None:
            result = add_position_to_detections(
                result,
                img_h=img_h,
                img_w=img_w,
                min_bbox_fraction=getattr(
                    self.config, "position_min_bbox_fraction", POSITION_GRID_THRESHOLD
                ),
            )
        return result


def add_position_to_detections(
    detections: dict,
    img_h: int,
    img_w: int,
    min_bbox_fraction: float = POSITION_GRID_THRESHOLD,
) -> dict:
    """Return a copy of `detections` augmented with a `detection_positions` list.

    Each element of `detection_positions` is the 3×3 grid label for the
    corresponding bbox in `detection_boxes`, or None if the bbox is
    too small (< min_bbox_fraction of image area) or zero-area.

    This helper is pure: the input dict is not mutated, all existing
    fields are preserved verbatim. Downstream consumers that read the
    original keys keep working.

    Args:
        detections: Detection dict produced by ObjectDetector._detect_single.
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        min_bbox_fraction: Minimum bbox area as fraction of image area
            to emit a label (default 0.05 = 5%).

    Returns:
        New dict containing all original keys plus `detection_positions`.

    Raises:
        ValueError: If parallel arrays in `detections` have inconsistent lengths.
    """
    boxes = detections.get("detection_boxes", [])
    # Cross-check parallel arrays (defensive: corrupt dicts should fail loud).
    parallel_keys = (
        "detection_scores",
        "detection_class_names",
        "detection_class_entities",
        "detection_class_labels",
    )
    lengths = {k: len(detections.get(k, [])) for k in parallel_keys}
    lengths["detection_boxes"] = len(boxes)
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        raise ValueError(f"Detection dict has inconsistent array lengths: {lengths}")

    positions: list[str | None] = [
        position_from_bbox(
            list(bbox), img_w=img_w, img_h=img_h, min_bbox_fraction=min_bbox_fraction
        )
        for bbox in boxes
    ]
    return {**detections, "detection_positions": positions}
