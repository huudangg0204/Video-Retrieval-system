"""Video and frame API routes for AIC-26."""
from __future__ import annotations
import os
from urllib.parse import unquote
from flask import Blueprint, request, jsonify, send_file, abort
from loguru import logger
from aic26.config import settings

bp = Blueprint('video', __name__)
IMG_ALLOWED_BASE = os.path.abspath(str(settings.keyframes_dir))
# Project root = directory that contains the "data/" folder.
# keyframes_dir = E:\...\aic26\data\keyframes
# data_dir      = E:\...\aic26\data
# PROJECT_ROOT  = E:\...\aic26   ← client paths like "data/keyframes/..." resolve from here
_PROJECT_ROOT = os.path.dirname(os.path.dirname(IMG_ALLOWED_BASE))

def _get_retriever():
    from flask import current_app
    r = current_app.config.get('AIC26_RETRIEVER')
    if r is None:
        fn = current_app.config.get('_get_retriever')
        if fn: r = fn(); current_app.config['AIC26_RETRIEVER'] = r
    return r

@bp.route('/get_frames/<path:video_name>', methods=['GET'])
def get_frames(video_name: str):
    r = _get_retriever()
    if r is None: return jsonify({'error': 'not initialized'}), 503
    return jsonify({'video_name': video_name, 'count': len(r.get_frames_of_video(video_name)), 'frames': r.get_frames_of_video(video_name)})

@bp.route('/get_frames_range/<path:video_name>/<int:frame_id>/<int:range_val>', methods=['GET'])
def get_frames_range(video_name: str, frame_id: int, range_val: int):
    r = _get_retriever()
    if r is None: return jsonify({'error': 'not initialized'}), 503
    frames = r.get_frames_of_video(video_name)
    s = max(0, frame_id - range_val)
    e = min(len(frames), frame_id + range_val + 1)
    return jsonify({'video_name': video_name, 'frames': frames[s:e]})

@bp.route('/image', methods=['GET'])
def serve_image():
    raw = request.args.get('path', '')
    if not raw: return abort(400)

    # Normalize slashes (handles Windows backslash from older paths)
    decoded = unquote(raw).replace('\\', '/')

    if os.path.isabs(decoded):
        # Already absolute (e.g. full Windows path stored in DB)
        abs_path = os.path.normpath(decoded)
    else:
        # Relative path like "data/keyframes/K01_V003/242.jpg"
        # Resolve from project root so we get:
        #   E:\...\aic26\data\keyframes\K01_V003\242.jpg
        rel = decoded.lstrip('/')
        abs_path = os.path.normpath(os.path.join(_PROJECT_ROOT, rel))

    logger.info(f"Requested: {abs_path}, exists: {os.path.exists(abs_path)}")
    if not os.path.exists(abs_path):
        return abort(404)
    return send_file(abs_path)