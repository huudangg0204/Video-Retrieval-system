"""
TRAKE API routes for AIC-26.

Endpoints for temporal retrieval of event sequences.
"""
from __future__ import annotations

import time

from flask import Blueprint, request, jsonify

from loguru import logger


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




bp = Blueprint('trake', __name__)


@bp.route('/trake', methods=['POST'])
def trake_search():
    """
    TRAKE: Temporal Retrieval and Alignment of Key Events.
    
    Request JSON:
        {
            "events": [
                "A player shooting basketball",
                "A player dunking*",
                "Crowd cheering"
            ],
            "top_k": 200,
            "candidates": 200,
            "option": "Highest"
        }
    
    Returns:
        JSON with elapsed time and TRAKE solutions
    """
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'error': 'retrieval engine not initialized'}), 503
    
    payload = request.get_json(force=True, silent=True) or {}
    events = payload.get('events', [])
    top_k = int(payload.get('top_k', 200) or 200)
    candidates = int(payload.get('candidates', 200) or 200)
    option = payload.get('option', 'Highest')
    
    if not events:
        return jsonify({'error': 'events required'}), 400
    
    logger.info(f"TRAKE search: {len(events)} events, option={option}")
    
    start = time.time()
    
    if option == 'Closest':
        results = retriever.trake_closest(
            events, top_k=top_k, candidates=candidates,
        )
    elif option == 'Highest':
        results = retriever.trake_highest(
            events, top_k=top_k, candidates=candidates,
        )
    elif option == 'All':
        return jsonify({'error': "option 'All' is not yet implemented"}), 400
    else:
        return jsonify({'error': f"Unknown option: {option!r}"}), 400
    
    elapsed = time.time() - start
    
    return jsonify({
        'elapsed': round(elapsed, 3),
        'count': len(results) if results else 0,
        'result': results,
    })