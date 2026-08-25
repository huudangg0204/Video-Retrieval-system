"""
Export API routes for AIC-26.

Endpoints for exporting search results and TRAKE solutions as CSV files.
"""
from __future__ import annotations

import os
import time

from flask import Blueprint, request, jsonify, send_from_directory


from aic26.config import settings


bp = Blueprint('export', __name__)


def _get_retriever():
    """Get retriever from current_app config.

    NOTE: this same helper is duplicated in system.py, video.py, and
    trake.py — it was missing here entirely, which meant every route in
    this file raised NameError: name '_get_retriever' is not defined the
    moment it was called (500 Internal Server Error on every export
    request). If you're touching this again, consider hoisting it into a
    shared aic26/api/routes/_common.py instead of a 4th copy-paste.
    """
    from flask import current_app
    retriever = current_app.config.get('AIC26_RETRIEVER')
    if retriever is None:
        get_retriever_fn = current_app.config.get('_get_retriever')
        if get_retriever_fn:
            retriever = get_retriever_fn()
            current_app.config['AIC26_RETRIEVER'] = retriever
    return retriever


@bp.route('/export_results', methods=['POST'])
def export_results():
    """
    Export search results to CSV submission file.
    
    Request JSON:
        {
            "results": [...],
            "name": "my_submission",
            "const_value": 1
        }
    
    Returns:
        JSON with download URL
    """
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'error': 'retrieval engine not initialized'}), 503
    
    payload = request.get_json(force=True, silent=True) or {}
    results = payload.get('results', [])
    name = payload.get('name', f"exported_{int(time.time())}")
    const_value = payload.get('const_value', None)
    if const_value in ('', None):
        const_value = None
    else:
        const_value = str(const_value).strip()
        if not const_value:
            const_value = None
    
    try:
        retriever.export_results(results, name, const_value)
    except ValueError as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 400
    
    filename = f"{name}.csv"
    
    return jsonify({
        'status': 'ok',
        'csv_path': filename,
        'download_url': f"/download/{filename}",
    })


@bp.route('/export_trake_results', methods=['POST'])
def export_trake_results():
    """
    Export TRAKE results to CSV submission file.
    
    Request JSON:
        {
            "results": [...],
            "name": "trake_submission"
        }
    
    Returns:
        JSON with download URL
    """
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'error': 'retrieval engine not initialized'}), 503
    
    payload = request.get_json(force=True, silent=True) or {}
    results = payload.get('results', [])
    name = payload.get('name', f"trake_{int(time.time())}")
    
    if not results:
        return jsonify({'status': 'error', 'msg': 'No results provided'}), 400
    
    try:
        os.makedirs(str(settings.output_dir), exist_ok=True)
        retriever.export_trake(results, name)
        filename = f"{name}.csv"
        
        return jsonify({
            'status': 'ok',
            'csv_path': filename,
            'download_url': f"/download/{filename}",
        })
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500


@bp.route('/export_submission', methods=['POST'])
def export_submission():
    """
    Generate a 20-row submission CSV centered on a single frame.
    
    Request JSON:
        {
            "video_name": "L21_V001",
            "frame_id": 1234,
            "gap": 15,
            "const_value": 1,
            "name": "my_sub"
        }
    
    Returns:
        JSON with CSV path
    """
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'error': 'retrieval engine not initialized'}), 503
    
    payload = request.get_json(force=True, silent=True) or {}
    video_name = payload.get('video_name')
    frame_id = payload.get('frame_id')
    gap = payload.get('gap')
    const_value = payload.get('const_value')
    name = payload.get('name', f"submission_{int(time.time())}")
    
    if const_value in ('', None):
        const_value = None
    else:
        const_value = str(const_value).strip()
        if not const_value:
            const_value = None
    
    if video_name is None or frame_id is None or gap is None:
        return jsonify({
            'status': 'error',
            'msg': 'video_name, frame_id, gap are required',
        }), 400
    
    try:
        csv_path = retriever.export_submission(
            video_name, frame_id, gap, const_value, name,
        )
        filename = os.path.basename(csv_path)
        return jsonify({
            'status': 'ok',
            'csv_path': csv_path,
            'download_url': f"/download/{filename}",
        })
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500


@bp.route('/export_trake_submission', methods=['POST'])
def export_trake_submission():
    """
    Generate a TRAKE sequence submission CSV from video name and frame list.
    
    Request JSON:
        {
            "video_name": "L26_V001",
            "frames": [12, 34, 56],  # or "12, 34, 56" / "12 34 56"
            "name": "my_trake_sub"
        }
    
    Returns:
        JSON with CSV path and download URL
    """
    import re
    
    retriever = _get_retriever()
    if retriever is None:
        return jsonify({'error': 'retrieval engine not initialized'}), 503
    
    payload = request.get_json(force=True, silent=True) or {}
    video_name = (payload.get('video_name') or '').strip()
    raw_frames = payload.get('frames', [])
    name = payload.get('name', f"trake_sub_{int(time.time())}")
    if not name or not str(name).strip():
        name = f"trake_sub_{int(time.time())}"
    
    if isinstance(raw_frames, str):
        frames = [int(f) for f in re.findall(r'\d+', raw_frames)]
    elif isinstance(raw_frames, list):
        frames = []
        for f in raw_frames:
            try:
                frames.append(int(f))
            except (ValueError, TypeError):
                continue
    else:
        frames = []
    
    if not video_name or not frames:
        return jsonify({
            'status': 'error',
            'msg': 'video_name and at least one frame are required',
        }), 400
    
    trake_payload = [
        {"video_name": video_name, "frame_idx": f, "event": len(frames)}
        for f in frames
    ]
    
    try:
        os.makedirs(str(settings.output_dir), exist_ok=True)
        csv_path = retriever.export_trake(trake_payload, name)
        filename = f"{name}.csv"
        return jsonify({
            'status': 'ok',
            'csv_path': csv_path,
            'download_url': f"/download/{filename}",
        })
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500