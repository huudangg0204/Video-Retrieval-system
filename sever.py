"""sever.py — Flask REST API for the EVA02 retrieval system.

Fixes vs. original:
1. export_trake_results: `filename` was never assigned → NameError in production. Fixed.
2. /api/trake option="All": called retriever.trake_all() which doesn't exist → now returns 400.
3. Consistent JSON error responses with HTTP status codes.
4. Upload dir created via config constant.
"""
import os
import time
import json

import numpy as np
from urllib.parse import unquote

from flask import Flask, request, jsonify, render_template, send_file, abort, send_from_directory
from flask_cors import CORS

from eva02_retrieval_trake import EVA02ImageRetrieval
from config import OUTPUT_FOLDER

# ── Server config ─────────────────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "0.0.0.0")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Initialise retriever (once, at startup) ───────────────────────────────────
print("🔁 Initializing EVA02 retriever (this may take a while)...")
retriever = EVA02ImageRetrieval()
print("✅ Retriever ready.")

IMG_ALLOWED_BASE = os.path.abspath(
    retriever.keyframes_dir if getattr(retriever, "keyframes_dir", None) else "./keyframes"
)
UPLOAD_DIR = os.path.abspath("./_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    stats = retriever.get_stats()
    return render_template("index.html", stats=stats)


# ── Search APIs ───────────────────────────────────────────────────────────────

@app.route("/api/search_text", methods=["POST"])
def api_search_text():
    payload   = request.get_json(force=True, silent=True) or {}
    query     = payload.get("query", "")
    top_k     = int(payload.get("top_k", 100) or 100)
    objects   = payload.get("objects", "")

    if not query:
        return jsonify({"error": "empty query"}), 400

    try:
        threshold = float(payload.get("threshold", 0.5))
    except (TypeError, ValueError):
        threshold = 0.5

    start   = time.time()
    results = retriever.search_text(query, top_k=top_k, save_to_db=True,
                                    objects=objects, threshold=threshold)
    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "results": results})


@app.route("/api/search_image", methods=["POST"])
def api_search_image():
    payload   = request.get_json(force=True, silent=True) or {}
    img_path  = payload.get("image_path", "")
    top_k     = int(payload.get("top_k", 100) or 100)

    if not img_path:
        return jsonify({"error": "image_path required"}), 400

    start   = time.time()
    results = retriever.search_image(img_path, top_k=top_k, save_to_db=True)
    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "results": results})


@app.route("/api/upload_image", methods=["POST"])
def api_upload_image():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400

    safe_name = f.filename.replace("/", "_").replace("\\", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{int(time.time() * 1000)}_{safe_name}")
    f.save(save_path)

    top_k   = int(request.form.get("top_k", 100) or 100)
    start   = time.time()
    results = retriever.search_image(save_path, top_k=top_k, save_to_db=True)
    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "results": results, "uploaded_path": save_path})


# ── TRAKE API ─────────────────────────────────────────────────────────────────

@app.route("/api/trake", methods=["POST"])
def api_trake():
    payload    = request.get_json(force=True, silent=True) or {}
    events     = payload.get("events", [])
    top_k      = int(payload.get("top_k",      200) or 200)
    candidates = int(payload.get("candidates", 200) or 200)
    option     = payload.get("option", "Closest")

    if not events:
        return jsonify({"error": "events required"}), 400

    start = time.time()

    if option == "Closest":
        results = retriever.trake_closest(events, top_k=top_k, candidates=candidates)
    elif option == "Highest":
        results = retriever.trake_highest(events, top_k=top_k, candidates=candidates)
    elif option == "All":
        # trake_all is not implemented — return a clear error instead of crashing
        return jsonify({"error": "option 'All' is not yet implemented"}), 400
    else:
        return jsonify({"error": f"Unknown option: {option!r}"}), 400

    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "result": results})


# ── Stored queries ────────────────────────────────────────────────────────────

@app.route("/api/stored_queries", methods=["GET"])
def api_stored_queries():
    return jsonify(retriever.get_stored_queries())


@app.route("/api/reload_query/<int:idx>", methods=["GET"])
def api_reload_query(idx):
    return jsonify(retriever.reload_query_results(idx))


# ── Video frames ──────────────────────────────────────────────────────────────

@app.route("/api/get_frames/<path:video_name>", methods=["GET"])
def api_get_frames(video_name):
    return jsonify(retriever.get_frames_of_video(video_name))


@app.route("/api/get_frames_range/<path:video_name>/<int:frame_id>/<int:range_val>", methods=["GET"])
def api_get_frames_range(video_name, frame_id, range_val):
    frames    = retriever.get_frames_of_video(video_name)
    start_idx = max(0, frame_id - range_val)
    end_idx   = min(len(frames), frame_id + range_val + 1)
    return jsonify(frames[start_idx:end_idx])


# ── Export APIs ───────────────────────────────────────────────────────────────

@app.route("/api/export_results", methods=["POST"])
def api_export_results():
    payload     = request.get_json(force=True, silent=True) or {}
    results     = payload.get("results", [])
    name        = payload.get("name", f"exported_{int(time.time())}")
    const_value = payload.get("const_value", None)

    if const_value in ("", None):
        const_value = None
    else:
        try:
            const_value = int(const_value)
        except (TypeError, ValueError):
            const_value = None

    try:
        retriever.export_results(results, name, const_value)
    except ValueError as e:
        return jsonify({"status": "error", "msg": str(e)}), 400

    filename = f"{name}.csv"
    return jsonify({
        "status":       "ok",
        "csv_path":     filename,
        "download_url": f"/download/{filename}",
    })


@app.route("/api/export_trake_results", methods=["POST"])
def api_export_trake_results():
    """
    Export a flat list of TRAKE result rows to a submission CSV.

    FIX: original code referenced undefined `filename` variable → NameError.
    Now we build `filename` from `name` consistently.
    """
    payload = request.get_json(force=True, silent=True) or {}
    results = payload.get("results", [])
    name    = payload.get("name", f"trake_{int(time.time())}")

    if not results:
        return jsonify({"status": "error", "msg": "No results provided"}), 400

    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        retriever.export_trake(results, name)
        filename = f"{name}.csv"   # ← was missing in original, causing NameError
        return jsonify({
            "status":       "ok",
            "csv_path":     filename,
            "download_url": f"/download/{filename}",
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


@app.route("/api/export_submission", methods=["POST"])
def api_export_submission():
    payload     = request.get_json(force=True, silent=True) or {}
    video_name  = payload.get("video_name")
    frame_id    = payload.get("frame_id")
    gap         = payload.get("gap")
    const_value = payload.get("const_value")
    name        = payload.get("name", f"submission_{int(time.time())}")

    if video_name is None or frame_id is None or gap is None:
        return jsonify({"status": "error", "msg": "video_name, frame_id, gap are required"}), 400

    try:
        csv_path = retriever.export_submission(video_name, frame_id, gap, const_value, name)
        return jsonify({"status": "ok", "csv_path": csv_path})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


# ── Grouping API ──────────────────────────────────────────────────────────────

@app.route("/api/group_by_video", methods=["POST"])
def api_group_by_video():
    payload       = request.get_json(force=True, silent=True) or {}
    results       = payload.get("results", [])
    sort_by       = payload.get("sort_by", "frame")
    top_per_video = payload.get("top_per_video", None)

    if isinstance(top_per_video, str) and top_per_video.strip():
        try:
            top_per_video = int(top_per_video)
        except ValueError:
            top_per_video = None

    grouped = retriever.group_results_by_video(
        results, sort_by=sort_by, top_per_video=top_per_video, with_time=True
    )
    return jsonify(grouped)


# ── Combined text+image search ────────────────────────────────────────────────

@app.route("/api/search_text_with_image", methods=["POST"])
def api_search_text_with_image():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400

    query     = request.form.get("query", "")
    top_k     = int(request.form.get("top_k", 100) or 100)
    objects   = request.form.get("objects", "")
    threshold = float(request.form.get("threshold", 0.5))

    if not query:
        return jsonify({"error": "empty query"}), 400

    safe_name = f.filename.replace("/", "_").replace("\\", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{int(time.time() * 1000)}_{safe_name}")
    f.save(save_path)

    start = time.time()

    text_results = retriever.search_text(
        query, top_k=top_k * 2, save_to_db=False, objects=objects, threshold=threshold
    )

    if not text_results:
        _safe_remove(save_path)
        return jsonify({"error": "no text results found"}), 404

    query_features = retriever.encode_image(save_path)
    reranked: list = []

    for candidate in text_results:
        candidate_features = retriever.get_embedding(candidate["image_path"])
        if candidate_features is None:
            continue
        q_f32 = query_features.astype(np.float32)
        c_f32 = candidate_features.astype(np.float32)
        dino_sim = float(
            np.dot(q_f32[0], c_f32[0]) /
            (np.linalg.norm(q_f32[0]) * np.linalg.norm(c_f32[0]) + 1e-9)
        )
        combined = 0.6 * candidate["similarity"] + 0.4 * dino_sim
        copy = dict(candidate)
        copy["similarity"] = float(combined)
        reranked.append(copy)

    reranked.sort(key=lambda x: x["similarity"], reverse=True)
    final_results = reranked[:top_k]
    elapsed = time.time() - start

    _safe_remove(save_path)
    return jsonify({"elapsed": float(elapsed), "results": final_results})


# ── Image serving ─────────────────────────────────────────────────────────────

@app.route("/image")
def serve_image():
    raw = request.args.get("path", "")
    if not raw:
        return abort(400, "path param required")

    decoded  = unquote(raw)
    abs_path = os.path.abspath(decoded)

    try:
        common = os.path.commonpath([abs_path, IMG_ALLOWED_BASE])
    except ValueError:
        return abort(403)

    if common != IMG_ALLOWED_BASE and not abs_path.startswith(IMG_ALLOWED_BASE):
        return abort(403)

    if not os.path.exists(abs_path):
        return abort(404)

    return send_file(abs_path)


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def api_stats():
    return jsonify(retriever.get_stats())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🌐 Starting server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)