"""
Search API routes for AIC-26.

Endpoints for text search, image search, and combined search.
"""
from __future__ import annotations

import time
import os
from pathlib import Path

from loguru import logger
import json

# ── In-memory metadata cache (PA4) ─────────────────────────────
_META_CACHE: dict[str, dict] | None = None


def _get_meta_cache() -> dict[str, dict]:
    """Return the shared metadata cache.

    Tries to reuse the cache that BM25CaptionIndex already loaded from
    the pre-built ``_cache.pkl``.  Falls back to loading the pickle
    directly, or scanning individual JSON files as a last resort.
    """
    global _META_CACHE
    if _META_CACHE is not None:
        return _META_CACHE

    # Try to grab the cache from the retriever's BM25 index (already loaded)
    try:
        from flask import current_app
        retriever = current_app.config.get('AIC26_RETRIEVER')
        if retriever is not None:
            bm25_idx = getattr(retriever, '_searcher', None)
            if bm25_idx is not None:
                bm25_idx = getattr(bm25_idx, 'bm25_index', None)
            if bm25_idx is not None and getattr(bm25_idx, '_meta_cache', None):
                _META_CACHE = bm25_idx._meta_cache
                logger.info(f"Metadata cache shared from BM25: {len(_META_CACHE)} entries")
                return _META_CACHE
    except RuntimeError:
        pass  # Outside Flask request context

    # Fallback: load from pickle or scan JSON
    import pickle
    from aic26.config import settings
    cache_path = Path(settings.metadata_dir) / "_cache.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            _META_CACHE = pickle.load(f)
        logger.info(f"Metadata cache loaded from pickle: {len(_META_CACHE)} entries")
    else:
        # Last resort: read individual JSON files
        meta_dir = Path(settings.metadata_dir)
        cache: dict[str, dict] = {}
        if meta_dir.exists():
            for json_path in meta_dir.rglob("*.json"):
                key = f"{json_path.parent.name}/{json_path.stem}"
                try:
                    cache[key] = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
        _META_CACHE = cache
        logger.info(f"Metadata cache loaded (JSON scan): {len(_META_CACHE)} entries")

    return _META_CACHE


def _attach_metadata(results):
    cache = _get_meta_cache()
    for r in results:
        video = r.get("video_name")
        frame = r.get("frame_idx")
        r["ocr_text"] = ""
        r["scene"] = ""
        r["scene_group"] = ""
        r["caption"] = ""
        r["detections"] = []
        if video and frame:
            key = f"{video}/{frame}"
            meta = cache.get(key)
            if meta:
                r["ocr_text"] = meta.get("ocr_text", "")
                r["scene"] = meta.get("scene", "")
                r["scene_group"] = meta.get("scene_group", "")
                r["caption"] = meta.get("caption", "")
                r["detections"] = meta.get("detections", [])
    return results


def _apply_pre_filters(
    required_objects: list[str],
    object_mode: str,
    scene_group: str,
) -> set[str] | None:
    """Scan _META_CACHE and return a set of matching frame keys.

    Returns None when no filters are active (no restriction).
    Each key has the form ``"video_name/frame_idx"``.
    """
    has_objects = bool(required_objects)
    has_scene = bool(scene_group)
    if not has_objects and not has_scene:
        return None

    cache = _get_meta_cache()
    matched: set[str] = set()

    for key, meta in cache.items():
        # ── Scene group filter ──────────────────────────────────
        if has_scene:
            if meta.get("scene_group", "") != scene_group:
                continue

        # ── Object presence filter ─────────────────────────────
        if has_objects:
            detection_labels = [
                d.get("label", "").lower()
                for d in (meta.get("detections") or [])
            ]
            req_lower = [o.lower() for o in required_objects]
            if object_mode == "or":
                if not any(req in detection_labels for req in req_lower):
                    continue
            else:  # default: "and"
                if not all(req in detection_labels for req in req_lower):
                    continue

        matched.add(key)

    logger.info(
        f"Pre-filter: scene_group={scene_group!r}, "
        f"objects={required_objects} ({object_mode}) "
        f"→ {len(matched)} candidate frames"
    )
    return matched



def _get_retriever():
    """Get retriever from current_app config."""
    from flask import current_app
    retriever = current_app.config.get('AIC26_RETRIEVER')
    if retriever is None:
        get_retriever_fn = current_app.config.get('_get_retriever')
        if get_retriever_fn:
            retriever = get_retriever_fn()
            current_app.config['AIC26_RETRIEVER'] = retriever
    return retriever

import numpy as np
from flask import Blueprint, request, jsonify, current_app



# get_retriever via current_app

bp = Blueprint('search', __name__)


@bp.route('/search_text', methods=['POST'])
def search_text():
    """
    Search keyframes by text description.
    
    Request JSON:
        {
            "query": "A red car on the street",
            "top_k": 100,
            "objects": "car,person",
            "threshold": 0.5
        }
    
    Returns:
        JSON with elapsed time and list of result objects
    """
    payload = request.get_json(force=True, silent=True) or {}
    query = payload.get('query', '')

    if not query:
        return jsonify({'error': 'empty query'}), 400

    # 🔥 Tự động kiểm tra và chạy pipeline nếu cần
    if not _ensure_embeddings_loaded():
        return jsonify({
            'error': 'No embeddings found and no videos to process. '
                     'Please add videos to data/videos/ first.'
        }), 400

    # Reload retriever nếu cần
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'error': 'retrieval engine not initialized'}), 503

    top_k = int(payload.get('top_k', 100) or 100)
    objects = payload.get('objects', '')
    threshold = float(payload.get('threshold', 0.5))
    caption_weight_raw = payload.get('caption_weight', None)
    caption_weight = (
        float(caption_weight_raw) if caption_weight_raw is not None else None
    )

    # —— Pre-search filters (object presence + scene group) —————————
    required_objects: list[str] = payload.get('required_objects') or []
    object_mode: str = str(payload.get('object_mode', 'and')).lower()
    scene_group: str = str(payload.get('scene_group', '')).strip()

    candidate_ids: set[str] | None = _apply_pre_filters(
        required_objects=required_objects,
        object_mode=object_mode,
        scene_group=scene_group,
    )

    # Expand fetch_k when filters are active so we can intersect without
    # running out of results.
    fetch_k = top_k * 5 if candidate_ids is not None else top_k

    logger.info(f"Text search: '{query[:80]}...' (top_k={top_k}, fetch_k={fetch_k})")

    start = time.time()
    results = retriever.search_text(
        query=query,
        top_k=fetch_k,
        save_to_db=True,
        objects=objects,
        threshold=threshold,
    )

    # Apply candidate filter when active
    if candidate_ids is not None:
        results = [
            r for r in results
            if f"{r.get('video_name')}/{r.get('frame_idx')}" in candidate_ids
        ][:top_k]

    elapsed = time.time() - start

    results = _attach_metadata(results)

    return jsonify({
        'elapsed': round(elapsed, 3),
        'count': len(results),
        'results': results,
    })



@bp.route('/search_image', methods=['POST'])
def search_image():
    """
    Search keyframes by image similarity.
    
    Can accept either an image path or an uploaded file.
    
    Returns:
        JSON with elapsed time and list of result objects
    """
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'error': 'retrieval engine not initialized'}), 503
    
    # Check for file upload
    if 'file' in request.files:
        f = request.files['file']
        if not f.filename:
            return jsonify({'error': 'empty filename'}), 400
        
        upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
        upload_dir.mkdir(exist_ok=True)
        
        safe_name = Path(f.filename).name
        save_path = upload_dir / f"{int(time.time() * 1000)}_{safe_name}"
        f.save(str(save_path))
        image_path = str(save_path)
    else:
        payload = request.get_json(force=True, silent=True) or {}
        image_path = payload.get('image_path', '')
        if not image_path:
            return jsonify({'error': 'image_path or file required'}), 400
    
    top_k = int(request.form.get('top_k', 100) or 100)
    
    logger.info(f"Image search: {image_path} (top_k={top_k})")
    
    start = time.time()
    results = retriever.search_image(
        image_path=image_path,
        top_k=top_k,
        save_to_db=True,
    )
    elapsed = time.time() - start
    
    results = _attach_metadata(results)
    
    return jsonify({
        'elapsed': round(elapsed, 3),
        'count': len(results),
        'results': results,
    })


@bp.route('/upload_image', methods=['POST'])
def upload_and_search():
    """
    Upload an image and search by it.
    
    Accepts multipart form data with optional text query for combined search.
    
    Returns:
        JSON with elapsed time and list of result objects
    """
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'error': 'retrieval engine not initialized'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'empty filename'}), 400
    
    # Save uploaded file
    upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
    upload_dir.mkdir(exist_ok=True)
    
    safe_name = Path(f.filename).name
    save_path = upload_dir / f"{int(time.time() * 1000)}_{safe_name}"
    f.save(str(save_path))
    
    top_k = int(request.form.get('top_k', 100) or 100)
    query = request.form.get('query', '')
    objects = request.form.get('objects', '')
    threshold = float(request.form.get('threshold', 0.5))
    caption_weight_raw = request.form.get('caption_weight', None)
    caption_weight = (
        float(caption_weight_raw) if caption_weight_raw not in (None, '') else None
    )

    start = time.time()

    if query:
        # Combined text + image search
        text_results = retriever.search_text(
            query, top_k=top_k * 2, save_to_db=False,
            objects=objects, threshold=threshold,
        )
        
        if not text_results:
            _safe_remove(save_path)
            return jsonify({'error': 'no text results found'}), 404
        
        query_features = retriever.encode_image(str(save_path))
        reranked = []
        
        for candidate in text_results:
            candidate_features = retriever.get_embedding(candidate['image_path'])
            if candidate_features is None:
                continue
            
            q_f32 = query_features.astype(np.float32)
            c_f32 = candidate_features.astype(np.float32)
            
            dino_sim = float(
                np.dot(q_f32[0], c_f32[0]) /
                (np.linalg.norm(q_f32[0]) * np.linalg.norm(c_f32[0]) + 1e-9)
            )
            
            combined = 0.6 * candidate['similarity'] + 0.4 * dino_sim
            copy = dict(candidate)
            copy['similarity'] = float(combined)
            reranked.append(copy)
        
        reranked.sort(key=lambda x: x['similarity'], reverse=True)
        results = reranked[:top_k]
    else:
        # Image-only search
        results = retriever.search_image(
            str(save_path), top_k=top_k, save_to_db=True,
        )
    
    elapsed = time.time() - start
    _safe_remove(save_path)
    
    results = _attach_metadata(results)
    
    return jsonify({
        'elapsed': round(elapsed, 3),
        'count': len(results),
        'results': results,
        'uploaded_path': str(save_path),
    })

@bp.route('/search_text_with_image', methods=['POST'])
def search_text_with_image():
    """
    Combined text + image search.
    
    Accepts multipart form data with file and query parameters.
    """
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'error': 'retrieval engine not initialized'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'empty filename'}), 400
    
    query = request.form.get('query', '')
    top_k = int(request.form.get('top_k', 100) or 100)
    objects = request.form.get('objects', '')
    threshold = float(request.form.get('threshold', 0.5))
    caption_weight_raw = request.form.get('caption_weight', None)
    caption_weight = (
        float(caption_weight_raw) if caption_weight_raw not in (None, '') else None
    )

    if not query:
        return jsonify({'error': 'empty query'}), 400

    # Save uploaded file
    upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
    upload_dir.mkdir(exist_ok=True)

    safe_name = Path(f.filename).name
    save_path = upload_dir / f"{int(time.time() * 1000)}_{safe_name}"
    f.save(str(save_path))

    start = time.time()

    # Text search first
    text_results = retriever.search_text(
        query, top_k=top_k * 2, save_to_db=False,
        objects=objects, threshold=threshold,
    )
    
    if not text_results:
        _safe_remove(save_path)
        return jsonify({'error': 'no text results found'}), 404
    
    # Rerank with image similarity
    query_features = retriever.encode_image(str(save_path))
    reranked = []
    
    for candidate in text_results:
        candidate_features = retriever.get_embedding(candidate['image_path'])
        if candidate_features is None:
            continue
        
        q_f32 = query_features.astype(np.float32)
        c_f32 = candidate_features.astype(np.float32)
        
        dino_sim = float(
            np.dot(q_f32[0], c_f32[0]) /
            (np.linalg.norm(q_f32[0]) * np.linalg.norm(c_f32[0]) + 1e-9)
        )
        
        combined = 0.6 * candidate['similarity'] + 0.4 * dino_sim
        copy = dict(candidate)
        copy['similarity'] = float(combined)
        reranked.append(copy)
    
    reranked.sort(key=lambda x: x['similarity'], reverse=True)
    results = reranked[:top_k]
    elapsed = time.time() - start
    
    _safe_remove(save_path)
    
    results = _attach_metadata(results)
    
    return jsonify({
        'elapsed': round(elapsed, 3),
        'count': len(results),
        'results': results,
    })

def _safe_remove(path) -> None:
    """Safely remove a file."""
    try:
        if isinstance(path, Path):
            os.remove(str(path))
        else:
            os.remove(path)
    except OSError:
        pass


@bp.route('/search_ocr', methods=['POST'])
def search_ocr():
    """Search keyframes by OCR text content.

    Independent OCR block: scans the full metadata cache for frames
    whose ``ocr_text`` field matches the query string.

    Request JSON::

        {
            "query": "HANOI",
            "top_k": 50,
            "case_insensitive": true,
            "mode": "contains"   # "contains" | "exact" | "any_word"
        }

    Returns a list of result objects in the same format as other search
    endpoints so the frontend can reuse the same card renderer.
    """
    payload = request.get_json(force=True, silent=True) or {}
    query: str = str(payload.get('query', '')).strip()
    if not query:
        return jsonify({'error': 'empty query'}), 400

    top_k = int(payload.get('top_k', 50) or 50)
    case_insensitive: bool = bool(payload.get('case_insensitive', True))
    mode: str = str(payload.get('mode', 'contains')).lower()

    q = query.lower() if case_insensitive else query
    words = q.split() if mode == 'any_word' else []

    cache = _get_meta_cache()
    matched: list[dict] = []

    for key, meta in cache.items():
        ocr_entries = meta.get('ocr_text') or []
        # Flatten all OCR text for this frame into one string
        if isinstance(ocr_entries, list):
            frame_text = ' '.join(
                entry.get('text', '') if isinstance(entry, dict) else str(entry)
                for entry in ocr_entries
            )
        else:
            frame_text = str(ocr_entries)

        compare_text = frame_text.lower() if case_insensitive else frame_text

        hit = False
        if mode == 'exact':
            hit = compare_text == q
        elif mode == 'any_word':
            hit = any(w in compare_text for w in words)
        else:  # default: contains
            hit = q in compare_text

        if not hit:
            continue

        # Build result object consistent with other search endpoints
        parts = key.split('/', 1)
        video_name = parts[0] if len(parts) == 2 else ''
        frame_idx = parts[1] if len(parts) == 2 else ''

        from aic26.config import settings as _s
        img_path = str(
            Path(_s.keyframes_dir) / video_name / f"{frame_idx}.jpg"
        )

        matched.append({
            'video_name': video_name,
            'frame_idx': frame_idx,
            'image_path': img_path,
            'similarity': 1.0,
            'ocr_text': ocr_entries,
            'scene': meta.get('scene', ''),
            'scene_group': meta.get('scene_group', ''),
            'caption': meta.get('caption', ''),
            'detections': meta.get('detections', []),
        })

        if len(matched) >= top_k:
            break

    return jsonify({
        'count': len(matched),
        'results': matched,
    })

def _ensure_embeddings_loaded():
    """
    Kiểm tra embeddings đã có chưa. Nếu chưa, tự động chạy pipeline.
    Returns True nếu sẵn sàng search.
    """
    from pathlib import Path
    from aic26.config import settings
    
    embedding_dir = Path(settings.embedding_dir)
    npy_files = list(embedding_dir.glob("*.npy"))
    
    if npy_files:
        return True  # Đã có embeddings
    
    # Chưa có embeddings - kiểm tra video
    video_dir = Path("data/videos")
    if not video_dir.exists() or not list(video_dir.glob("*")):
        return False  # Không có video để xử lý
    
    # Tự động chạy pipeline
    from aic26.pipeline.orchestrator import PipelineOrchestrator
    
    orchestrator = PipelineOrchestrator(settings)
    orchestrator.run(
        video_dir=str(video_dir),
        skip_ocr=True,
        skip_detection=True,
    )
    
    # Reload retriever với embeddings mới
    from flask import current_app
    current_app.config['AIC26_RETRIEVER'] = None  # Reset để load lại
    
    return True