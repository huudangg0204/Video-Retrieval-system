"""
Error handling middleware for AIC-26 API.

Provides consistent JSON error responses and proper HTTP status codes.
"""
from __future__ import annotations

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from loguru import logger


def register_error_handlers(app: Flask) -> None:
    """
    Register error handlers for the Flask application.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request."""
        logger.warning(f"400 Bad Request: {request.url} - {error}")
        return jsonify({
            "error": "bad_request",
            "message": str(error) or "Bad request",
            "status": 400,
        }), 400
    
    @app.errorhandler(403)
    def forbidden(error):
        """Handle 403 Forbidden."""
        logger.warning(f"403 Forbidden: {request.url}")
        return jsonify({
            "error": "forbidden",
            "message": "Access denied",
            "status": 403,
        }), 403
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found."""
        return jsonify({
            "error": "not_found",
            "message": str(error) or "Resource not found",
            "status": 404,
        }), 404
    
    @app.errorhandler(413)
    def too_large(error):
        """Handle 413 Payload Too Large."""
        return jsonify({
            "error": "payload_too_large",
            "message": "Uploaded file exceeds size limit",
            "status": 413,
        }), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server Error."""
        logger.error(f"500 Internal Error: {error}")
        return jsonify({
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "status": 500,
        }), 500
    
    @app.errorhandler(Exception)
    def unhandled_error(error):
        """Handle unhandled exceptions."""
        logger.error(f"Unhandled error: {error}", exc_info=True)
        
        if isinstance(error, HTTPException):
            return jsonify({
                "error": "http_error",
                "message": str(error),
                "status": error.code,
            }), error.code
        
        return jsonify({
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "status": 500,
        }), 500