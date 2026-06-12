"""
System API routes for AIC-26.

Endpoints for system statistics, stored queries, and grouping.
"""
from __future__ import annotations

from flask import Blueprint, request, jsonify

from loguru import logger

bp = Blueprint('system', __name__)


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
