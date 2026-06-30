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

def _attach_metadata(results):
    from aic26.config import settings
    meta_dir = Path(settings.metadata_dir)
    for r in results:
        video = r.get("video_name")
        frame = r.get("frame_idx")
        r["ocr_text"] = ""
        r["scene"] = ""
        r["caption"] = ""
        r["detections"] = []
        if video and frame:
            json_path = meta_dir / video / f"{frame}.json"
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        r["ocr_text"] = meta.get("ocr_text", "")
                        r["scene"] = meta.get("scene", "")
                        r["caption"] = meta.get("caption", "")
                        r["detections"] = meta.get("detections", [])
                except Exception:
                    pass
    return results


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

    logger.info(f"Text search: '{query[:80]}...' (top_k={top_k})")

    start = time.time()
    results = retriever.search_text(
        query=query,
        top_k=top_k,
        save_to_db=True,
        objects=objects,
        threshold=threshold,
        caption_weight=caption_weight,
    )
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
            caption_weight=caption_weight,
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
        caption_weight=caption_weight,
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