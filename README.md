# AIC-26: AI Challenge Video Retrieval System

[![CI/CD](https://github.com/your-org/aic26/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/aic26/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue)](https://hub.docker.com/r/your-org/aic26)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Intelligent video retrieval system using EVA-02 CLIP embeddings, FAISS vector search, and TRAKE temporal reasoning.**

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
  - [CLI Commands](#cli-commands)
  - [Data Pipeline](#data-pipeline)
  - [Search](#search)
  - [API Server](#api-server)
  - [Desktop GUI](#desktop-gui)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Docker](#-docker)
- [Development](#-development)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Changelog](#-changelog)
- [License](#-license)

---

## 📊 Overview

AIC-26 is a comprehensive video retrieval system designed for AI challenges and research. It enables searching through large video datasets using natural language descriptions, images, or temporal event sequences.

### Core Technologies

| Technology | Purpose |
|------------|---------|
| **EVA-02 CLIP** | Text and image embedding (768-dim) |
| **FAISS** | Vector similarity search (IndexFlatIP) |
| **TRAKE** | Temporal Retrieval and Alignment of Key Events |
| **YOLOv8** | Object detection (Open Images V7, 600+ classes) |
| **Faster R-CNN** | COCO object detection (91 classes) |
| **BLIP ITM** | Image-text matching for result reranking |
| **PaddleOCR** | Vietnamese text extraction from frames |
| **DINOv2** | Keyframe extraction via scene change detection |
| **Flask** | REST API server |
| **CustomTkinter** | Desktop GUI |

### Use Cases

- 🔍 **Text-to-Video Search**: Find video segments matching natural language descriptions
- 🖼️ **Image-to-Video Search**: Find visually similar video frames
- ⏱️ **Temporal Event Search**: Find sequences of events in chronological order
- 🎯 **Object-Based Filtering**: Filter results by detected objects
- 📝 **OCR Search**: Search through text appearing in videos

---

## ✨ Features

### Multi-Modal Search
- **Text Search**: Describe what you're looking for in natural language
- **Image Search**: Upload an image to find visually similar frames
- **Combined Search**: Use both text and image for better accuracy
- **Multi-Query**: Run multiple text queries simultaneously

### TRAKE Temporal Reasoning
- **Sequence Finding**: Locate video segments where events occur in specific order
- **Two Strategies**:
  - `closest`: Minimize temporal distance between events
  - `highest`: Maximize total similarity score
- **BLIP Reranking**: Use `*` suffix on events for enhanced precision

### Data Processing Pipeline
Videos → Keyframes → Object Detection → OCR → Embeddings → FAISS Index
- **Automated Pipeline**: Single command to process everything
- **Resumable**: Checkpoint support for long-running tasks
- **Parallel Processing**: Multi-threaded for speed
- **GPU Accelerated**: CUDA support throughout

### Interfaces
- **REST API**: Flask-based with 16+ endpoints
- **Desktop GUI**: Full-featured CustomTkinter application
- **CLI**: Unified command-line interface
- **Web UI**: Built-in search interface

---

## 🏗 Architecture

### High-Level Architecture
┌─────────────────────────────────────────────────────────────────────┐
│ AIC-26 SYSTEM │
├─────────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ │
│ │ Data Pipeline │ │ Core Engine │ │ Interfaces │ │
│ ├──────────────────┤ ├──────────────────┤ ├──────────────────┤ │
│ │ Keyframe │ │ ModelRegistry │ │ REST API │ │
│ │ Extraction │ │ EmbeddingIndex │ │ Desktop GUI │ │
│ │ Object Detection │ │ VectorDatabase │ │ CLI Commands │ │
│ │ OCR Processing │ │ SearchEngine │ │ Web UI │ │
│ │ Embedding │ │ Reranker │ │ │ │
│ │ FAISS Indexing │ │ TrakeEngine │ │ │ │
│ │ │ │ Exporter │ │ │ │
│ └──────────────────┘ └──────────────────┘ └──────────────────┘ │
│ │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ Shared Utilities │ │
│ │ Logging | File Utils | Frame Utils | Media Utils │ │
│ └──────────────────────────────────────────────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────────┘

### Package Structure
aic26/
├── pyproject.toml # Project metadata & dependencies
├── README.md # This file
├── LICENSE # MIT License
├── CHANGELOG.md # Version history
├── CONTRIBUTING.md # Contribution guidelines
├── Makefile # Development shortcuts
├── .gitignore
├── .pre-commit-config.yaml
├── .env.example # Environment variables template
├── heroku.yml # Heroku deployment
├── Procfile # Heroku process file
├── mkdocs.yml # Documentation config
│
├── src/
│ └── aic26/ # Main package
│ ├── init.py
│ │
│ ├── config/ # Configuration layer
│ │ ├── init.py
│ │ ├── settings.py # Pydantic Settings
│ │ └── constants.py # COCO categories, etc.
│ │
│ ├── core/ # Core retrieval engine
│ │ ├── init.py
│ │ ├── models.py # ModelRegistry
│ │ ├── embeddings.py # EmbeddingIndex
│ │ ├── database.py # VectorDatabase
│ │ ├── search.py # SearchEngine
│ │ ├── rerank.py # BLIP Reranker
│ │ ├── trake.py # TrakeEngine
│ │ ├── export.py # CSV Exporter
│ │ └── retrieval.py # EVA02ImageRetrieval (Facade)
│ │
│ ├── pipeline/ # Data processing
│ │ ├── init.py
│ │ ├── keyframe.py # KeyframeExtractor
│ │ ├── detection.py # ObjectDetector
│ │ ├── ocr.py # OCRExtractor
│ │ ├── embedding.py # EmbeddingExtractor
│ │ ├── indexer.py # IndexBuilder
│ │ └── orchestrator.py # PipelineOrchestrator
│ │
│ ├── api/ # REST API
│ │ ├── init.py
│ │ ├── server.py # Flask app factory
│ │ ├── middleware/
│ │ │ └── error_handler.py # Error handling
│ │ ├── routes/
│ │ │ ├── search.py # Search endpoints
│ │ │ ├── trake.py # TRAKE endpoints
│ │ │ ├── export.py # Export endpoints
│ │ │ ├── video.py # Video/frames endpoints
│ │ │ └── system.py # System/stats endpoints
│ │ └── templates/
│ │ └── index.html # Web UI
│ │
│ ├── gui/ # Desktop GUI
│ │ ├── init.py
│ │ ├── retrieval_app.py # Main search GUI
│ │ └── filter_app.py # Object filter GUI
│ │
│ ├── cli/ # CLI commands
│ │ ├── init.py
│ │ └── main.py # All CLI commands
│ │
│ └── utils/ # Utilities
│ ├── init.py
│ ├── logging.py # Loguru setup
│ ├── file_utils.py # File operations
│ ├── frame_utils.py # Frame parsing
│ └── media_utils.py # Video info
│
├── tests/ # Test suite
│ ├── conftest.py # Shared fixtures
│ └── unit/
│ ├── test_utils.py
│ ├── test_config.py
│ ├── test_core.py
│ └── test_pipeline.py
│
├── docker/ # Docker files
│ ├── Dockerfile # Production build
│ ├── Dockerfile.gpu # GPU-enabled build
│ ├── docker-compose.yml # Multi-service config
│ └── .dockerignore
│
├── configs/ # YAML configuration
│ ├── pipeline.yaml # Pipeline parameters
│ └── api_config.yaml # API settings
│
├── scripts/ # Utility scripts
│ ├── download_videos.py # Video downloader
│ ├── setup_db.py # Database initializer
│ └── migrate_data.py # Data migration tool
│
├── docs/ # Documentation
│ └── index.md
│
└── .github/ # CI/CD
└── workflows/
├── ci.yml # Main CI pipeline
└── docs.yml # Documentation deploy

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (optional, but recommended)
- 16GB+ RAM
- 50GB+ free disk space (for video data and embeddings)

### One-Command Setup

```bash
# Clone and install
git clone https://github.com/your-org/aic26.git
cd aic26
pip install -e .

# Verify installation
aic26 check
# Start the API server
aic26 serve api

# Open browser at http://localhost:5000
# Use the built-in Web UI to search