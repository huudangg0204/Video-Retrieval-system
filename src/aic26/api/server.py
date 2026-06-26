"""
Flask application factory for AIC-26 API.

Creates and configures the Flask application with all routes,
middleware, and the retrieval engine instance.

Migrated from: sever.py
"""
from __future__ import annotations

import os
from pathlib import Path

from loguru import logger
from flask import Flask, render_template
from flask_cors import CORS


from aic26.config import settings
from aic26.api.middleware.error_handler import register_error_handlers
from aic26.api.routes import search, trake, export, video, system


def create_app(config=None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Settings instance (uses global settings if None)
        
    Returns:
        Configured Flask application
    """
    if config is None:
        config = settings
    
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    
    # ── Configuration ──────────────────────────────────────────
    app.config.from_mapping(
        SECRET_KEY=config.flask_secret_key,
        UPLOAD_FOLDER=str(config.upload_dir),
        MAX_CONTENT_LENGTH=config.max_upload_size_mb * 1024 * 1024,
        AIC26_SETTINGS=config,
    )
    
    # Ensure upload directory exists
    Path(config.upload_dir).mkdir(parents=True, exist_ok=True)
    
    # ── Extensions ─────────────────────────────────────────────
    CORS(app)
    
    # ── Initialize retriever (lazy) ────────────────────────────
    _init_retriever(app, config)
    
    # ── Register blueprints ────────────────────────────────────
    app.register_blueprint(search.bp, url_prefix='/api')
    app.register_blueprint(trake.bp, url_prefix='/api')
    app.register_blueprint(export.bp, url_prefix='/api')
    app.register_blueprint(video.bp, url_prefix='/api')
    app.register_blueprint(system.bp, url_prefix='/api')
    
    # ── Error handlers ─────────────────────────────────────────
    register_error_handlers(app)
    
    # ── Main route ─────────────────────────────────────────────
    @app.route('/')
    def index():
        """Main page with search interface."""
        retriever = app.config.get('AIC26_RETRIEVER')
        if retriever:
            stats = retriever.get_stats()
        else:
            stats = {"total_images": 0, "total_videos": 0}
        return render_template('index.html', stats=stats)
    
    # ── Health check ───────────────────────────────────────────
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return {"status": "ok", "version": "1.0.0"}
    
    logger.info("Flask application created successfully.")
    
    @app.route('/download/<path:filename>')
    def download_file(filename: str):
        """Download an exported CSV file."""
        from flask import send_from_directory
        return send_from_directory(
            str(config.output_dir),
            filename,
            as_attachment=True,
        )
    return app
    


def _init_retriever(app: Flask, config) -> None:
    """
    Initialize the EVA02ImageRetrieval instance eagerly at startup.

    All models, embeddings, and the FAISS index are loaded immediately so
    the server is fully ready before handling any request.
    """
    logger.info("Initializing retrieval engine at startup...")
    from aic26.core.retrieval import EVA02ImageRetrieval

    retriever_instance = EVA02ImageRetrieval(
        embedding_dir=str(config.embedding_dir),
        keyframes_dir=str(config.keyframes_dir),
        db_path=str(config.db_path),
    )
    app.config['AIC26_RETRIEVER'] = retriever_instance
    logger.info("Retrieval engine ready.")


def get_retriever(app: Flask = None):
    """
    Get the retrieval engine instance.
    
    Args:
        app: Flask application instance
        
    Returns:
        EVA02ImageRetrieval instance
    """
    if app is None:
        from flask import current_app
        app = current_app
    
    retriever = app.config.get('AIC26_RETRIEVER')
    if retriever is None:
        get_retriever_fn = app.config.get('_get_retriever')
        if get_retriever_fn:
            retriever = get_retriever_fn()
    
    return retriever