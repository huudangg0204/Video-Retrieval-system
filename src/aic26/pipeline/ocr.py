"""
OCR text extraction from keyframes using PaddleOCR.

Extracts text from keyframe images and saves results as CSV
with checkpoint support for resumable processing.

Migrated from: get_ocr_header.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from loguru import logger

from aic26.config import Settings
from aic26.utils.file_utils import ensure_dir


class OCRExtractor:
    """
    Extract text from keyframe images using PaddleOCR.
    
    Features:
    - Vietnamese language support
    - Angle classification for rotated text
    - Checkpoint-based resumable processing
    - Confidence filtering
    
    Usage:
        ocr = OCRExtractor(settings)
        ocr.process("./keyframes", "./ocr_results.csv")
    """
    
    def __init__(self, config: Optional['Settings'] = None) -> None:
        """
        Initialize the OCR extractor.
        
        Args:
            config: Settings instance
        """
        if config is None:
            from aic26.config import settings as global_settings
            config = global_settings
        
        self.config = config
        self._ocr = None
    
    def _check_paddle_stability(self) -> bool:
        """Test if PaddleOCR crashes (e.g., SIGABRT/SIGSEGV) on this hardware."""
        import multiprocessing
        
        def _test_worker():
            try:
                import logging
                logging.getLogger("ppocr").setLevel(logging.ERROR)
                from paddleocr import PaddleOCR
                import numpy as np
                ocr = PaddleOCR(lang='vi', use_gpu=True, show_log=False)
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                ocr.ocr(dummy)
            except Exception:
                pass
                
        p = multiprocessing.Process(target=_test_worker)
        p.start()
        p.join()
        
        return p.exitcode == 0

    def _load_ocr(self):
        """Lazy-load OCR model with fallback logic."""
        if self._ocr is None:
            # 1. Test PaddleOCR stability
            paddle_stable = False
            if os.environ.get("FORCE_EASYOCR") != "1":
                logger.info("Checking PaddleOCR stability on this hardware...")
                paddle_stable = self._check_paddle_stability()
            
            # 2. Try PaddleOCR if stable
            if paddle_stable:
                try:
                    from paddleocr import PaddleOCR
                    logger.info("PaddleOCR is stable. Loading PaddleOCR model...")
                    self._ocr = PaddleOCR(
                        lang='vi',
                        use_angle_cls=True,
                        det_limit_side_len=1280,
                        rec_batch_num=64,
                        show_log=False,
                        use_gpu=True,
                        ir_optim=False,
                    )
                    self.engine = 'paddle'
                    logger.info("PaddleOCR model ready.")
                    return
                except Exception as e:
                    logger.warning(f"PaddleOCR initialization failed: {e}. Falling back to EasyOCR.")
            else:
                logger.warning("PaddleOCR is unstable (SIGABRT/SIGSEGV detected) or FORCE_EASYOCR=1. Falling back to EasyOCR.")
            
            # 3. Fallback to EasyOCR
            try:
                import easyocr
                logger.info("Loading EasyOCR model...")
                self._ocr = easyocr.Reader(['vi', 'en'], gpu=True)
                self.engine = 'easyocr'
                logger.info("EasyOCR model ready.")
            except ImportError:
                logger.error("EasyOCR not installed. Install with: pip install easyocr")
                raise
    
    def process(
        self,
        input_dir: str,
        output_csv: Optional[str] = None,
        annotate: bool = False,
        ann_dir: Optional[str] = None,
        min_conf: float = 0.35,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Process all keyframe folders for OCR.
        
        Args:
            input_dir: Directory containing video subfolders with keyframes
            output_csv: Path to output CSV file
            annotate: Whether to save annotated images
            ann_dir: Directory for annotated images
            min_conf: Minimum confidence threshold
            limit: Limit number of folders to process
            
        Returns:
            DataFrame with OCR results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_csv = output_csv or str(self.config.ocr_output_path)
        
        self._load_ocr()
        
        if annotate:
            ann_dir = ann_dir or os.path.join(str(input_path), "annotations")
            ensure_dir(ann_dir)
        
        # Checkpoint directory
        checkpoint_dir = os.path.join(str(input_path), "ocr_checkpoints")
        ensure_dir(checkpoint_dir)
        
        # Find video folders
        folders = sorted(
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
            and d not in ("annotations", "ocr_checkpoints")
        )
        
        if limit:
            folders = folders[:limit]
        
        logger.info(f"Processing {len(folders)} folder(s) for OCR...")
        
        all_rows = []
        
        for folder in tqdm(folders, desc="OCR folders", unit="folder"):
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{folder}_{Path(output_csv).name}",
            )
            
            # Check for existing checkpoint
            if os.path.exists(checkpoint_path):
                logger.debug(f"Loading checkpoint for {folder}")
                try:
                    checkpoint_df = pd.read_csv(checkpoint_path)
                    all_rows.extend(checkpoint_df.to_dict(orient="records"))
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}")
            
            folder_path = os.path.join(input_dir, folder)
            folder_rows = self._process_folder(
                folder_path,
                folder,
                min_conf,
                annotate,
                ann_dir,
            )
            
            all_rows.extend(folder_rows)
            
            # Save checkpoint
            if folder_rows:
                try:
                    pd.DataFrame(folder_rows).to_csv(
                        checkpoint_path,
                        index=False,
                        encoding="utf-8",
                    )
                except Exception as e:
                    logger.warning(f"Failed to save checkpoint: {e}")
        
        # Save full CSV
        df = pd.DataFrame(all_rows)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        
        logger.info(
            f"OCR complete: {len(all_rows)} text boxes in {len(df)} images → {output_csv}"
        )
        
        return df
    
    def _process_folder(
        self,
        folder_path: str,
        folder_name: str,
        min_conf: float,
        annotate: bool,
        ann_dir: Optional[str],
    ) -> List[Dict]:
        """
        Process all images in a single folder.
        
        Args:
            folder_path: Path to folder containing images
            folder_name: Name of the folder (video name)
            min_conf: Minimum confidence threshold
            annotate: Whether to save annotated images
            ann_dir: Directory for annotated images
            
        Returns:
            List of row dicts
        """
        image_files = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        
        folder_rows = []
        
        for img_file in tqdm(
            image_files,
            desc=f"  {folder_name}",
            unit="img",
            leave=False,
        ):
            img_path = os.path.join(folder_path, img_file)
            
            # Skip very small files
            if os.path.getsize(img_path) < 1024:
                logger.debug(f"Skipping small image: {img_file}")
                continue
            
            try:
                items = self._detect_text(img_path, min_conf)
                
                row = {
                    "video": folder_name,
                    "file": img_file,
                    "path": img_path,
                    "text_full": " ".join(it["text"] for it in items),
                    "num_boxes": len(items),
                    "conf_mean": float(np.mean([it["conf"] for it in items])) if items else None,
                    "boxes": json.dumps([it["box"] for it in items], ensure_ascii=False),
                    "texts": json.dumps([it["text"] for it in items], ensure_ascii=False),
                    "confs": json.dumps([it["conf"] for it in items], ensure_ascii=False),
                }
                
                folder_rows.append(row)
                
                # Annotate if requested
                if annotate and ann_dir:
                    sub_ann = os.path.join(ann_dir, folder_name)
                    ensure_dir(sub_ann)
                    out_img = os.path.join(sub_ann, img_file)
                    self._draw_text_boxes(img_path, items, out_img)
                    
            except Exception as e:
                logger.warning(f"Error processing {img_file}: {e}")
        
        return folder_rows
    
    def _detect_text(
        self,
        img_path: str,
        min_conf: float = 0.30,
    ) -> List[Dict]:
        """
        Detect all text in an image.
        
        Args:
            img_path: Path to image file
            min_conf: Minimum confidence threshold
            
        Returns:
            List of dicts with 'text', 'conf', 'box' keys
        """
        items = []
        
        if getattr(self, 'engine', 'paddle') == 'paddle':
            bgr = cv2.imread(img_path)
            if bgr is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                res = self._ocr.ocr(rgb, cls=True)
            except Exception as e:
                logger.warning(f"PaddleOCR inference error: {e}")
                return []
                
            if res is None or not len(res):
                return []
            
            if isinstance(res[0], list):
                res = res[0]
                
            for r in res:
                if r is None or len(r) < 2:
                    continue
                box, (txt, conf) = r
                c = float(conf)
                if c >= min_conf:
                    items.append({
                        "text": txt,
                        "conf": c,
                        "box": [[float(x), float(y)] for (x, y) in box],
                    })
        else:
            # EasyOCR accepts paths directly
            res = self._ocr.readtext(img_path)
            
            if res is None or len(res) == 0:
                return []
            
            for r in res:
                if len(r) < 3:
                    continue
                box, txt, conf = r
                c = float(conf)
                if c >= min_conf:
                    items.append({
                        "text": txt,
                        "conf": c,
                        "box": [[float(x), float(y)] for (x, y) in box],
                    })
        
        # Sort by line position for readability
        items.sort(
            key=lambda it: (
                round(min(p[1] for p in it["box"]) / 10),
                min(p[0] for p in it["box"]),
            )
        )
        
        return items
    
    @staticmethod
    def _draw_text_boxes(
        img_path: str,
        items: List[Dict],
        out_path: str,
        show_text: bool = True,
    ) -> str:
        """
        Draw bounding boxes and text on an image.
        
        Args:
            img_path: Path to input image
            items: List of detection items
            out_path: Path to output image
            show_text: Whether to display text content
            
        Returns:
            Path to output image
        """
        bgr = cv2.imread(img_path)
        canvas = bgr.copy()
        
        for it in items:
            pts = np.array(it["box"], dtype=np.int32)
            cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
            
            if show_text:
                txt = f'{it["text"]} ({it["conf"]:.2f})'
                x = int(min(p[0] for p in it["box"]))
                y = int(min(p[1] for p in it["box"])) - 6
                cv2.putText(
                    canvas, txt[:60],
                    (x, max(y, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA,
                )
        
        cv2.imwrite(out_path, canvas)
        return out_path