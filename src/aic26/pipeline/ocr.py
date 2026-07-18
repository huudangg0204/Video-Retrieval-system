"""
OCR text extraction from keyframes using PaddleOCR.

Extracts text from keyframe images and saves results as CSV
with checkpoint support for resumable processing.

Parallelization modes:
- image-level (default): nhiều worker xử lý từng ảnh trong 1 folder → dùng khi chỉ có 1 folder
- folder-level: mỗi worker xử lý 1 folder → dùng khi có nhiều folder

Migrated from: get_ocr_header.py
"""
from __future__ import annotations

import json
import os
import multiprocessing
import warnings
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from loguru import logger

from aic26.config import Settings
from aic26.utils.file_utils import ensure_dir

# ── Global state cho worker ──────────────────────────────────────────────────
_global_ocr = None
_global_min_conf = 0.35


def _suppress_paddle_logs():
    """Tắt toàn bộ log/warning của Paddle và PaddleOCR."""
    import logging
    os.environ["PADDLE_CPP_LOG_LEVEL"] = "3"
    os.environ["FLAGS_call_stack_level"] = "0"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.filterwarnings("ignore")
    for name in ("ppocr", "paddle", "paddlex", "paddle.fluid"):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.disable(logging.WARNING)


def _worker_init(min_conf: float):
    """Khởi tạo OCR model 1 lần duy nhất cho mỗi worker process."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    global _global_ocr, _global_min_conf
    _global_min_conf = min_conf

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    _suppress_paddle_logs()

    import cv2
    cv2.setNumThreads(0)

    from paddleocr import PaddleOCR
    _global_ocr = PaddleOCR(
        lang="vi",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=960,
        text_recognition_batch_size=32,
        device="cpu",
        enable_mkldnn=False,
    )


# ── Worker xử lý từng ẢNH (image-level parallelism) ─────────────────────────

def _worker_process_image(args):
    """
    Xử lý 1 ảnh đơn lẻ. Dùng cho image-level parallelism.
    Trả về dict row hoặc None nếu không có text.
    """
    img_path, img_file, folder_name, annotate, ann_dir = args
    min_conf = _global_min_conf

    try:
        if os.path.getsize(img_path) < 1024:
            return None

        bgr = cv2.imread(img_path)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        results = list(_global_ocr.predict(rgb))
        if not results:
            return None

        items = []
        for result in results:
            boxes  = result.get("dt_polys",   [])
            texts  = result.get("rec_texts",  [])
            scores = result.get("rec_scores", [])
            for box, txt, conf in zip(boxes, texts, scores):
                c = float(conf)
                if c < min_conf:
                    continue
                box_list = [[float(pt[0]), float(pt[1])] for pt in box]
                items.append({"text": str(txt), "conf": c, "box": box_list})

        if not items:
            return None

        items.sort(
            key=lambda it: (
                round(min(p[1] for p in it["box"]) / 10),
                min(p[0] for p in it["box"]),
            )
        )

        row = {
            "video": folder_name,
            "file": img_file,
            "path": img_path,
            "text_full": " ".join(it["text"] for it in items),
            "num_boxes": len(items),
            "conf_mean": float(np.mean([it["conf"] for it in items])),
            "boxes": json.dumps([it["box"] for it in items], ensure_ascii=False),
            "texts": json.dumps([it["text"] for it in items], ensure_ascii=False),
            "confs": json.dumps([it["conf"] for it in items], ensure_ascii=False),
        }

        # Annotate nếu cần
        if annotate and ann_dir:
            sub_ann = os.path.join(ann_dir, folder_name)
            os.makedirs(sub_ann, exist_ok=True)
            canvas = bgr.copy()
            for it in items:
                pts = np.array(it["box"], dtype=np.int32)
                cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
                label = f'{it["text"]} ({it["conf"]:.2f})'
                x = int(min(p[0] for p in it["box"]))
                y = int(min(p[1] for p in it["box"])) - 6
                cv2.putText(
                    canvas, label[:60], (x, max(y, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
                )
            cv2.imwrite(os.path.join(sub_ann, img_file), canvas)

        return row

    except Exception:
        return None


# ── Worker xử lý theo FOLDER (folder-level parallelism, giữ nguyên cũ) ──────

def _worker_process_folder(args):
    """Hàm worker xử lý 1 thư mục video. Dùng cho folder-level parallelism."""
    folder_path, folder_name, checkpoint_path, annotate, ann_dir = args
    min_conf = _global_min_conf

    # Kiểm tra checkpoint
    if os.path.exists(checkpoint_path):
        try:
            cp_df = pd.read_csv(checkpoint_path)
            if len(cp_df) > 0:
                return cp_df.to_dict(orient="records")
        except Exception:
            pass

    image_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )

    rows = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        row = _worker_process_image((img_path, img_file, folder_name, annotate, ann_dir))
        if row:
            rows.append(row)

    # Lưu checkpoint
    try:
        if rows:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False, encoding="utf-8")
        else:
            pd.DataFrame([{"video": folder_name}]).to_csv(checkpoint_path, index=False, encoding="utf-8")
    except Exception:
        pass

    return rows


# ── OCRExtractor ─────────────────────────────────────────────────────────────

class OCRExtractor:
    """
    Extract text from keyframe images using PaddleOCR with Multiprocessing.

    Features:
    - Vietnamese language support
    - Checkpoint-based resumable processing
    - Confidence filtering
    - 2 chế độ parallel:
        * parallel_mode="image" (default): N workers xử lý từng ảnh song song
          → Tốt khi chỉ có 1 folder hoặc muốn tận dụng CPU tối đa cho 1 folder
        * parallel_mode="folder": mỗi worker xử lý 1 folder
          → Tốt khi có nhiều folder (hành vi cũ)

    Usage:
        # Xử lý 1 folder với 4 workers song song theo ảnh:
        ocr = OCRExtractor(settings)
        ocr.process("./keyframes/K11_V001", "./ocr_results.csv", parallel_mode="image", num_workers=4)

        # Xử lý nhiều folder, mỗi worker 1 folder:
        ocr.process("./keyframes", "./ocr_results.csv", parallel_mode="folder", num_workers=4)
    """

    def __init__(self, config: Optional['Settings'] = None) -> None:
        if config is None:
            from aic26.config import settings as global_settings
            config = global_settings
        self.config = config

    def process(
        self,
        input_dir: Optional[str] = None,
        output_csv: Optional[str] = None,
        annotate: bool = False,
        ann_dir: Optional[str] = None,
        min_conf: float = 0.35,
        limit: Optional[int] = None,
        num_workers: int = 2,
        parallel_mode: str = "image",   # "image" hoặc "folder"
        checkpoint_per_image: int = 50,  # ghi CSV sau mỗi N ảnh (chỉ dùng ở image mode)
    ) -> pd.DataFrame:
        """
        Process keyframe images for OCR.

        Args:
            input_dir: Thư mục chứa keyframes. Có thể là:
                       - Thư mục gốc chứa nhiều folder con (folder mode)
                       - Thư mục 1 folder ảnh trực tiếp (image mode tự detect)
            output_csv: Đường dẫn file CSV output.
            annotate: Vẽ bounding box lên ảnh gốc và lưu vào ann_dir.
            ann_dir: Thư mục lưu ảnh annotated.
            min_conf: Ngưỡng confidence tối thiểu (0.0–1.0).
            limit: Giới hạn số folder xử lý (chỉ dùng ở folder mode).
            num_workers: Số worker process song song.
            parallel_mode: "image" = parallel theo ảnh, "folder" = parallel theo folder.
            checkpoint_per_image: Tần suất ghi CSV tạm (chỉ image mode).
        """
        input_dir = input_dir or str(self.config.keyframes_dir)
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_csv = output_csv or str(self.config.ocr_output_path)
        out_csv_path = Path(output_csv)

        if annotate:
            ann_dir = ann_dir or str(input_path / "annotations")
            ensure_dir(ann_dir)

        # ── Auto-detect mode: nếu input_dir chứa ảnh trực tiếp → image mode ─
        has_images_directly = any(
            f.lower().endswith(('.jpg', '.jpeg', '.png'))
            for f in os.listdir(input_dir)
        )
        if has_images_directly and parallel_mode == "folder":
            logger.warning(
                "Phát hiện ảnh trực tiếp trong input_dir, tự động chuyển sang parallel_mode='image'"
            )
            parallel_mode = "image"

        _suppress_paddle_logs()

        if parallel_mode == "image":
            return self._process_image_parallel(
                input_dir=input_dir,
                input_path=input_path,
                output_csv=output_csv,
                out_csv_path=out_csv_path,
                annotate=annotate,
                ann_dir=ann_dir,
                min_conf=min_conf,
                num_workers=num_workers,
                checkpoint_per_image=checkpoint_per_image,
            )
        else:
            return self._process_folder_parallel(
                input_dir=input_dir,
                input_path=input_path,
                output_csv=output_csv,
                out_csv_path=out_csv_path,
                annotate=annotate,
                ann_dir=ann_dir,
                min_conf=min_conf,
                limit=limit,
                num_workers=num_workers,
            )

    # ── Image-level parallel ──────────────────────────────────────────────────

    def _process_image_parallel(
        self,
        input_dir, input_path, output_csv, out_csv_path,
        annotate, ann_dir, min_conf, num_workers, checkpoint_per_image,
    ) -> pd.DataFrame:
        """N workers xử lý từng ảnh song song trong 1 hoặc nhiều folder."""

        # Thu thập tất cả ảnh cần xử lý
        # Hỗ trợ cả: input_dir là 1 folder ảnh, hoặc là folder gốc chứa subfolder
        all_image_tasks = []

        entries = os.listdir(input_dir)
        subfolders = [
            e for e in entries
            if os.path.isdir(os.path.join(input_dir, e))
            and e not in ("annotations", "ocr_checkpoints", "checkpoints")
        ]

        if subfolders:
            # Có subfolder → xử lý nhiều folder với image parallelism
            for folder_name in sorted(subfolders):
                folder_path = os.path.join(input_dir, folder_name)
                for img_file in sorted(os.listdir(folder_path)):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_image_tasks.append((
                            os.path.join(folder_path, img_file),
                            img_file,
                            folder_name,
                            annotate,
                            ann_dir,
                        ))
        else:
            # Không có subfolder → input_dir chính là folder ảnh
            folder_name = input_path.name
            for img_file in sorted(entries):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_image_tasks.append((
                        os.path.join(input_dir, img_file),
                        img_file,
                        folder_name,
                        annotate,
                        ann_dir,
                    ))

        if not all_image_tasks:
            logger.info("Không tìm thấy ảnh nào để xử lý.")
            return pd.DataFrame()

        # Lọc ảnh đã có trong checkpoint CSV (resumable)
        done_paths = set()
        if out_csv_path.exists():
            try:
                existing = pd.read_csv(output_csv)
                if "path" in existing.columns:
                    done_paths = set(existing["path"].dropna().tolist())
                else:
                    # Nếu file mất header, cột thứ 3 (index 2) thường là cột path
                    done_paths = set(existing.iloc[:, 2].dropna().tolist())
                logger.info(f"Bỏ qua {len(done_paths)} ảnh đã xử lý (từ checkpoint CSV).")
            except Exception as e:
                logger.warning(f"Lỗi đọc checkpoint: {e}")

        pending_tasks = [t for t in all_image_tasks if t[0] not in done_paths]

        if not pending_tasks:
            logger.info("Tất cả ảnh đã được xử lý.")
            return pd.read_csv(output_csv) if out_csv_path.exists() else pd.DataFrame()

        actual_workers = max(1, min(num_workers, len(pending_tasks), multiprocessing.cpu_count()))
        logger.info(
            f"[Image-parallel] {len(pending_tasks)} ảnh còn lại "
            f"/ {len(all_image_tasks)} tổng | {actual_workers} worker(s)"
        )

        ctx = multiprocessing.get_context("spawn")
        all_rows = []
        buffer = []

        with ctx.Pool(
            processes=actual_workers,
            initializer=_worker_init,
            initargs=(min_conf,),
        ) as pool:
            try:
                for row in tqdm(
                    pool.imap_unordered(_worker_process_image, pending_tasks),
                    total=len(pending_tasks),
                    desc="OCR images",
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                ):
                    if row is None:
                        continue
    
                    all_rows.append(row)
                    buffer.append(row)
    
                    # Ghi CSV định kỳ để resumable
                    if len(buffer) >= checkpoint_per_image:
                        try:
                            buf_df = pd.DataFrame(buffer)
                            write_header = not out_csv_path.exists()
                            buf_df.to_csv(output_csv, mode="a", header=write_header, index=False, encoding="utf-8")
                            buffer.clear()
                        except Exception as e:
                            logger.warning(f"Lỗi ghi CSV tạm: {e}")
            except KeyboardInterrupt:
                logger.warning("Đã hủy quá trình OCR (Ctrl+C). Đang đóng worker an toàn...")
                pool.terminate()
                pool.join()

        # Flush buffer còn lại
        if buffer:
            try:
                buf_df = pd.DataFrame(buffer)
                write_header = not out_csv_path.exists()
                buf_df.to_csv(output_csv, mode="a", header=write_header, index=False, encoding="utf-8")
            except Exception as e:
                logger.warning(f"Lỗi ghi CSV cuối: {e}")

        # Gộp với existing và ghi lại sạch (loại header thừa)
        final_rows = []
        if out_csv_path.exists():
            try:
                final_rows = pd.read_csv(output_csv).to_dict(orient="records")
            except Exception:
                final_rows = all_rows

        df = pd.DataFrame(final_rows)
        df.to_csv(output_csv, index=False, encoding="utf-8")

        logger.info(f"OCR complete: {len(all_rows)} ảnh mới → tổng {len(df)} ảnh → {output_csv}")
        return df

    # ── Folder-level parallel (giữ nguyên hành vi cũ) ────────────────────────

    def _process_folder_parallel(
        self,
        input_dir, input_path, output_csv, out_csv_path,
        annotate, ann_dir, min_conf, limit, num_workers,
    ) -> pd.DataFrame:
        """Mỗi worker xử lý 1 folder (hành vi cũ)."""

        checkpoint_dir = os.path.join(str(input_path), "ocr_checkpoints")
        ensure_dir(checkpoint_dir)

        folders = sorted(
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
            and d not in ("annotations", "ocr_checkpoints", "checkpoints")
        )

        if limit:
            folders = folders[:limit]

        tasks = [
            (
                os.path.join(input_dir, folder),
                folder,
                os.path.join(checkpoint_dir, f"{folder}_{Path(output_csv).name}"),
                annotate,
                ann_dir,
            )
            for folder in folders
        ]

        if not tasks:
            logger.info("No folders found to process.")
            return pd.DataFrame()

        actual_workers = max(1, min(num_workers, len(tasks), multiprocessing.cpu_count()))
        logger.info(f"[Folder-parallel] {len(folders)} folder(s) | {actual_workers} worker(s)")

        all_rows = []
        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(
            processes=actual_workers,
            initializer=_worker_init,
            initargs=(min_conf,),
        ) as pool:
            try:
                for rows_batch in tqdm(
                    pool.imap_unordered(_worker_process_folder, tasks),
                    total=len(tasks),
                    desc="OCR folders",
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                ):
                    if not rows_batch:
                        continue
                    valid_rows = [r for r in rows_batch if "path" in r]
                    if not valid_rows:
                        continue
    
                    all_rows.extend(valid_rows)
    
                    try:
                        folder_df = pd.DataFrame(valid_rows)
                        write_header = not out_csv_path.exists()
                        folder_df.to_csv(output_csv, mode="a", header=write_header, index=False, encoding="utf-8")
                    except Exception as e:
                        logger.warning(f"Lỗi append CSV: {e}")
            except KeyboardInterrupt:
                logger.warning("Đã hủy quá trình OCR (Ctrl+C). Đang đóng worker an toàn...")
                pool.terminate()
                pool.join()

        df = pd.DataFrame(all_rows)
        df.to_csv(output_csv, index=False, encoding="utf-8")

        logger.info(f"OCR complete: {len(all_rows)} ảnh → {output_csv}")
        return df