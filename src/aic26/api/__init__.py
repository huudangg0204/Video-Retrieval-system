"""
REST API layer for AIC-26.

Provides Flask-based REST API endpoints for:
- Text and image search
- TRAKE temporal search
- Video frame browsing
- Result export
- System statistics

Usage:
    from aic26.api.server import create_app
    app = create_app(settings)
    app.run(host="0.0.0.0", port=5000)
"""

from aic26.api.server import create_app

__all__ = ["create_app"]