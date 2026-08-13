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


def _attach_metadata_to_frames(frames: list[dict]) -> list[dict]:
    """Enrich frame dicts with metadata from the shared _META_CACHE."""
    try:
        from aic26.api.routes.search import _get_meta_cache
        cache = _get_meta_cache()
    except Exception:
        return frames
    for f in frames:
        video = f.get('video_name', '')
        frame = f.get('frame_idx', '')
        key = f"{video}/{frame}"
        meta = cache.get(key) if cache else None
        f['ocr_text'] = meta.get('ocr_text', '') if meta else ''
        f['scene'] = meta.get('scene', '') if meta else ''
        f['scene_group'] = meta.get('scene_group', '') if meta else ''
        f['caption'] = meta.get('caption', '') if meta else ''
        f['detections'] = meta.get('detections', []) if meta else []
    return frames


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
    context_frames = frames[s:e]
    context_frames = _attach_metadata_to_frames(context_frames)
    return jsonify({
        'video_name': video_name,
        'center_frame_id': frame_id,
        'frames': context_frames,
    })

@bp.route('/browse/folder/<path:video_name>', methods=['GET'])
def browse_folder(video_name: str):
    """Return all keyframes for a video with full metadata.

    Used by the Browse Block → Folder View tab.
    ``video_name`` must match the folder name under ``data/keyframes/``
    (e.g. ``L26_V001``).
    """
    r = _get_retriever()
    if r is None:
        return jsonify({'error': 'not initialized'}), 503
    frames = r.get_frames_of_video(video_name)
    if not frames:
        return jsonify({'error': f'No frames found for video: {video_name}'}), 404
    frames = _attach_metadata_to_frames(frames)
    return jsonify({
        'video_name': video_name,
        'count': len(frames),
        'frames': frames,
    })

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

    # ── Security: reject any path that escapes the project root ──────────
    # os.path.normpath alone does not stop ".." segments from resolving
    # outside _PROJECT_ROOT (e.g. "../../../../etc/passwd"), and an
    # absolute path can point anywhere on disk. Only serve files that
    # resolve to somewhere under the project root.
    real_path = os.path.realpath(abs_path)
    real_root = os.path.realpath(_PROJECT_ROOT)
    try:
        inside_root = os.path.commonpath([real_path, real_root]) == real_root
    except ValueError:
        # Different drives on Windows (e.g. requested path on D:, root on E:)
        inside_root = False
    if not inside_root:
        logger.warning(f"[SECURITY] /image blocked (outside project root): requested={raw!r} resolved={real_path}")
        return abort(403)

    logger.info(f"Requested: {abs_path}, exists: {os.path.exists(abs_path)}")
    if not os.path.exists(abs_path):
        return abort(404)
    return send_file(abs_path)