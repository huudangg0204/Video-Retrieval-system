"""
System API routes for AIC-26.

Endpoints for system statistics, stored queries, and grouping.
"""
from __future__ import annotations

import os
import threading
from flask import Blueprint, request, jsonify

from loguru import logger

bp = Blueprint('system', __name__)

# ── Background FAISS build state ─────────────────────────────────────────────
_faiss_build_state: dict = {"status": "idle", "message": "", "result": None}
_faiss_build_lock = threading.Lock()


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


@bp.route('/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics."""
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({
            'total_images': 0,
            'total_videos': 0,
            'embedding_shape': None,
            'status': 'not_initialized',
        })
    
    stats = retriever.get_stats()
    stats['status'] = 'ready'
    return jsonify(stats)


@bp.route('/stored_queries', methods=['GET'])
def get_stored_queries():
    """Get list of all stored queries."""
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'queries': []})
    
    queries = retriever.get_stored_queries()
    return jsonify({
        'count': len(queries),
        'queries': queries,
    })


@bp.route('/reload_query/<int:idx>', methods=['GET'])
def reload_query(idx: int):
    """Reload results from a stored query."""
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'error': 'retrieval engine not initialized'}), 503
    
    results = retriever.reload_query_results(idx)
    
    return jsonify({
        'query_index': idx,
        'count': len(results),
        'results': results,
    })


@bp.route('/group_by_video', methods=['POST'])
def group_by_video():
    """Group search results by video with timing enrichment."""
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'error': 'retrieval engine not initialized'}), 503
    
    payload = request.get_json(force=True, silent=True) or {}
    results = payload.get('results', [])
    sort_by = payload.get('sort_by', 'frame')
    top_per_video = payload.get('top_per_video', None)
    
    if isinstance(top_per_video, str) and top_per_video.strip():
        try:
            top_per_video = int(top_per_video)
        except ValueError:
            top_per_video = None
    
    grouped = retriever.group_results_by_video(
        results,
        sort_by=sort_by,
        top_per_video=top_per_video,
        with_time=True,
    )
    
    return jsonify(grouped)


@bp.route('/build_faiss_index', methods=['POST'])
def build_faiss_index():
    """
    Trigger a FAISS index rebuild from current embeddings.

    Runs IndexBuilder in a background thread so the request returns immediately.
    Poll /api/build_faiss_index/status to track progress.
    """
    global _faiss_build_state

    with _faiss_build_lock:
        if _faiss_build_state["status"] == "running":
            return jsonify({"status": "running", "message": "Build already in progress"}), 409

        _faiss_build_state = {"status": "running", "message": "Starting...", "result": None}

    def _run():
        global _faiss_build_state
        try:
            from aic26.config import settings
            from aic26.pipeline.indexer import IndexBuilder
            builder = IndexBuilder()

            # Rebuild the SAME index shape the live retriever is configured
            # for. Calling build() (single-stream) while
            # use_dual_embedding=true would silently drop the CLIP-32
            # stream from search until someone reruns the dual build by hand.
            clip32_dir = str(getattr(settings, "clip32_embedding_dir", ""))
            use_dual = bool(getattr(settings, "use_dual_embedding", False))

            if use_dual and clip32_dir and os.path.isdir(clip32_dir) and os.listdir(clip32_dir):
                logger.info("Dual-stream enabled: building EVA-02 + CLIP-32 index")
                result = builder.build_dual(
                    eva02_dir=str(settings.embedding_dir),
                    clip32_dir=clip32_dir,
                )
            else:
                if use_dual:
                    logger.warning(
                        f"[INDEX-BUILD] use_dual_embedding=true but clip32_embedding_dir "
                        f"({clip32_dir}) missing/empty — built single-stream EVA-02 index instead"
                    )
                result = builder.build()

            with _faiss_build_lock:
                _faiss_build_state = {
                    "status": "done",
                    "message": (
                        f"Built {result['embeddings_loaded']} vectors "
                        f"({result['file_size_mb']:.1f} MB) → {result['output_path']}"
                    ),
                    "result": result,
                }
            logger.info(f"FAISS build finished: {result}")
        except Exception as exc:
            logger.error(f"FAISS build failed: {exc}")
            with _faiss_build_lock:
                _faiss_build_state = {
                    "status": "error",
                    "message": str(exc),
                    "result": None,
                }

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "running", "message": "FAISS build started in background"})


@bp.route('/build_faiss_index/status', methods=['GET'])
def build_faiss_index_status():
    """Return current status of the FAISS build job."""
    with _faiss_build_lock:
        return jsonify(dict(_faiss_build_state))