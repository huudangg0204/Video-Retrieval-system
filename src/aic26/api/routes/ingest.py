"""
API route for data ingestion — upload video files or keyframe ZIP.

Endpoints:
  POST /api/ingest/upload   — Upload file, start background ingestion
  GET  /api/ingest/status   — Poll ingestion progress
  POST /api/ingest/cancel   — Cancel running ingestion
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, jsonify, request, current_app
from loguru import logger
from werkzeug.utils import secure_filename

bp = Blueprint("ingest", __name__)

# ── In-memory job registry (single-process, suitable for dev) ─────────────────
_jobs: Dict[str, Dict[str, Any]] = {}
_job_lock = threading.Lock()

ALLOWED_EXTENSIONS = {
    # Videos
    ".mp4", ".avi", ".mkv", ".mov", ".webm",
    # Keyframe archives
    ".zip",
}

MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _make_job(job_id: str, filename: str, mode: str) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "filename": filename,
        "mode": mode,           # "videos" | "keyframes"
        "status": "queued",     # queued | running | done | error | cancelled
        "progress": 0,
        "message": "Queued",
        "stats": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
    }


# ── POST /api/ingest/upload ───────────────────────────────────────────────────

@bp.route("/ingest/upload", methods=["POST"])
def ingest_upload():
    """
    Upload a video file or keyframe ZIP and start background ingestion.

    Form fields:
        file          — The file to ingest (.mp4/.avi/.mkv/.zip)
        skip_detection — "true"|"false" (default: false)
        skip_ocr       — "true"|"false" (default: false)
        skip_es        — "true"|"false" (default: false)
        force_recreate — "true"|"false" (default: false)
        no_blip        — "true"|"false" (default: false)
        batch_size     — int (default: 40)

    Returns:
        {"job_id": "...", "status": "queued", "filename": "..."}
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    if not _allowed(file.filename):
        return jsonify({
            "error": f"File type not supported. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    # Options
    def bool_field(name: str, default: bool = False) -> bool:
        val = request.form.get(name, "").lower()
        return val in ("true", "1", "yes") if val else default

    opts = {
        "skip_detection": bool_field("skip_detection", False),
        "skip_ocr":       bool_field("skip_ocr", False),
        "skip_es":        bool_field("skip_es", False),
        "force_recreate": bool_field("force_recreate", False),
        "encode_blip":    not bool_field("no_blip", False),
        "batch_size":     int(request.form.get("batch_size", 40)),
    }

    # Save uploaded file
    config = current_app.config.get("AIC26_SETTINGS")
    upload_dir = Path(config.upload_dir if config else "data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(file.filename)
    dest = upload_dir / safe_name
    file.save(str(dest))

    # Detect mode from extension
    ext = Path(safe_name).suffix.lower()
    mode = "keyframes" if ext == ".zip" else "videos"

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job = _make_job(job_id, safe_name, mode)

    with _job_lock:
        _jobs[job_id] = job

    # Start background thread
    thread = threading.Thread(
        target=_run_ingest_job,
        args=(job_id, str(dest), mode, opts),
        daemon=True,
        name=f"ingest-{job_id}",
    )
    thread.start()

    logger.info(f"Ingestion job {job_id} started: {safe_name} ({mode})")

    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "filename": safe_name,
        "mode": mode,
    })


# ── GET /api/ingest/status/<job_id> ──────────────────────────────────────────

@bp.route("/ingest/status/<job_id>", methods=["GET"])
def ingest_status(job_id: str):
    """Poll ingestion job status."""
    with _job_lock:
        job = _jobs.get(job_id)

    if job is None:
        return jsonify({"error": f"Job '{job_id}' not found"}), 404

    return jsonify(job)


# ── GET /api/ingest/jobs ──────────────────────────────────────────────────────

@bp.route("/ingest/jobs", methods=["GET"])
def ingest_jobs():
    """List all ingestion jobs (most recent first)."""
    with _job_lock:
        jobs = list(reversed(list(_jobs.values())))
    return jsonify(jobs)


# ── POST /api/ingest/cancel/<job_id> ─────────────────────────────────────────

@bp.route("/ingest/cancel/<job_id>", methods=["POST"])
def ingest_cancel(job_id: str):
    """Mark a queued job as cancelled (best-effort)."""
    with _job_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404
        if job["status"] in ("queued", "running"):
            job["status"] = "cancelled"
            job["message"] = "Cancelled by user"

    return jsonify({"job_id": job_id, "status": "cancelled"})


# ── Background worker ─────────────────────────────────────────────────────────

def _update_job(job_id: str, **kwargs) -> None:
    with _job_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _run_ingest_job(job_id: str, file_path: str, mode: str, opts: dict) -> None:
    """Background thread: run IngestPipeline and update job state."""
    _update_job(job_id,
                status="running",
                started_at=time.time(),
                message="Starting ingestion pipeline...")

    try:
        # Check if cancelled before heavy work
        with _job_lock:
            if _jobs.get(job_id, {}).get("status") == "cancelled":
                return

        from aic26.pipeline.ingest import IngestPipeline

        pipeline = IngestPipeline(
            skip_detection=opts["skip_detection"],
            skip_ocr=opts["skip_ocr"],
            skip_es_index=opts["skip_es"],
            force_es_recreate=opts["force_recreate"],
            encode_blip=opts["encode_blip"],
            batch_size=opts["batch_size"],
        )

        _update_job(job_id, progress=10, message="Resolving input...")

        if mode == "keyframes":
            _update_job(job_id, progress=15, message="Extracting keyframe ZIP...")
            stats = pipeline.run_from_keyframes(input_path=file_path)
        else:
            _update_job(job_id, progress=15, message="Extracting keyframes from video...")
            stats = pipeline.run_from_videos(video_dir=str(Path(file_path).parent))

        _update_job(job_id,
                    status="done",
                    progress=100,
                    finished_at=time.time(),
                    message=f"Done — {stats.es_indexed} frames indexed",
                    stats={
                        "videos_found": stats.videos_found,
                        "keyframes_found": stats.keyframes_found,
                        "embeddings_extracted": stats.embeddings_extracted,
                        "es_indexed": stats.es_indexed,
                        "es_failed": stats.es_failed,
                        "duration_seconds": round(stats.duration_seconds, 1),
                        "steps_completed": stats.steps_completed,
                        "steps_failed": stats.steps_failed,
                        "steps_skipped": stats.steps_skipped,
                    })

        logger.info(f"Job {job_id} completed: {stats}")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        _update_job(job_id,
                    status="error",
                    progress=0,
                    finished_at=time.time(),
                    message=f"Error: {str(e)[:200]}",
                    error=str(e))
