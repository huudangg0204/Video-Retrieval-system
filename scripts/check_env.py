#!/usr/bin/env python3
"""
check_env.py — Kiểm tra môi trường và dependencies trước khi chạy AIC-26.

Usage:
    py -3.10 scripts/check_env.py
"""
from __future__ import annotations

import sys
import importlib
import subprocess
from pathlib import Path


def ok(msg: str) -> None:
    print(f"  [OK]  {msg}")

def warn(msg: str) -> None:
    print(f"  [!!]  {msg}")

def fail(msg: str) -> None:
    print(f"  [XX]  {msg}")


def check_python() -> None:
    print("\n[Python]")
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} — requires 3.10+")


def check_packages() -> None:
    print("\n[Packages]")
    required = [
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("flask", "Flask"),
        ("elasticsearch", "elasticsearch-py"),
        ("loguru", "loguru"),
        ("click", "click"),
        ("pydantic", "pydantic"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("timm", "timm"),
        ("transformers", "transformers"),
    ]
    optional = [
        ("paddleocr", "PaddleOCR (OCR only)"),
    ]
    for mod, label in required:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{label} ({ver})")
        except ImportError:
            fail(f"{label} — NOT INSTALLED  →  pip install {label}")
    for mod, label in optional:
        try:
            importlib.import_module(mod)
            ok(f"{label} (installed)")
        except ImportError:
            warn(f"{label} — not installed (optional)")


def check_cuda() -> None:
    print("\n[CUDA / GPU]")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"CUDA available — {name} ({vram_gb:.1f} GB VRAM)")
            if vram_gb < 3.5:
                warn("VRAM < 3.5 GB — may run out of memory with dual embedding")
        else:
            warn("CUDA not available — running on CPU (much slower)")
    except Exception as e:
        fail(f"CUDA check error: {e}")


def check_elasticsearch() -> None:
    print("\n[Elasticsearch]")
    try:
        import requests
        r = requests.get("http://localhost:9200", timeout=3)
        if r.status_code == 200:
            info = r.json()
            ver = info.get("version", {}).get("number", "?")
            ok(f"Elasticsearch running — version {ver}")
            # Check index
            r2 = requests.get("http://localhost:9200/aic26_keyframes", timeout=3)
            if r2.status_code == 200:
                count_r = requests.get("http://localhost:9200/aic26_keyframes/_count", timeout=3)
                count = count_r.json().get("count", 0)
                ok(f"Index 'aic26_keyframes' exists — {count} documents")
            else:
                warn("Index 'aic26_keyframes' not found — run: aic26 pipeline init-es-index")
        else:
            fail(f"Elasticsearch returned HTTP {r.status_code}")
    except Exception:
        fail("Elasticsearch not reachable at localhost:9200")
        warn("Start with: docker compose -f docker/docker-compose.yml up elasticsearch -d")


def check_docker() -> None:
    print("\n[Docker]")
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            ok("Docker is running")
            # Check elasticsearch container
            ps = subprocess.run(
                ["docker", "ps", "--filter", "name=elasticsearch", "--format", "{{.Names}} ({{.Status}})"],
                capture_output=True, text=True, timeout=5
            )
            containers = ps.stdout.strip()
            if containers:
                ok(f"Container: {containers}")
            else:
                warn("Elasticsearch container not running")
        else:
            warn("Docker is not running or not accessible")
    except FileNotFoundError:
        warn("Docker not installed or not in PATH")
    except Exception as e:
        warn(f"Docker check error: {e}")


def check_paths() -> None:
    print("\n[Data Paths]")
    root = Path(__file__).parent.parent
    paths = [
        ("data/", "Data root"),
        ("data/keyframes/", "Keyframes"),
        ("data/embeddings/", "Embeddings"),
        ("data/uploads/", "Upload cache"),
    ]
    for rel, label in paths:
        p = root / rel
        if p.exists():
            count = sum(1 for _ in p.rglob("*") if _.is_file()) if p.is_dir() else 0
            ok(f"{label}: {p} ({count} files)")
        else:
            warn(f"{label}: {p} — not found (will be created on first use)")


if __name__ == "__main__":
    print("=" * 52)
    print("  AIC-26 Environment Check")
    print("=" * 52)
    check_python()
    check_packages()
    check_cuda()
    check_docker()
    check_elasticsearch()
    check_paths()
    print("\n" + "=" * 52)
    print("  Done. Fix any [XX] items before running the server.")
    print("=" * 52 + "\n")
