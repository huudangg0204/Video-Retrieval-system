"""
Smoke test for AIC-26 — run this BEFORE committing to a full,
multi-hour pipeline run on the competition dataset.

Catches broken dependencies, wrong model names/paths, and silently
mis-wired components in minutes instead of discovering them after
processing 100 videos. Each check is independent and prints PASS/FAIL/SKIP
so a single failure doesn't stop the rest from reporting.

Usage:
    python scripts/smoke_test.py                 # all checks
    python scripts/smoke_test.py --skip-heavy     # skip GPU/model-loading checks
    python scripts/smoke_test.py --only imports,settings,paths

Exit code is non-zero if any check FAILs (SKIP does not count as failure).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field


@dataclass
class Results:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def ok(self, name: str) -> None:
        self.passed.append(name)
        print(f"  ✅ PASS  {name}")

    def fail(self, name: str, reason: str) -> None:
        self.failed.append((name, reason))
        print(f"  ❌ FAIL  {name} — {reason}")

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append((name, reason))
        print(f"  ⏭️  SKIP  {name} — {reason}")


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


# ── 1. Imports ──────────────────────────────────────────────────────────

def check_imports(r: Results) -> None:
    section("Dependency imports")
    required = [
        "torch", "torchvision", "cv2", "numpy", "pandas",
        "faiss", "flask", "flask_cors", "open_clip", "transformers",
        "ultralytics", "loguru", "click", "pydantic", "pydantic_settings",
        "rank_bm25",
    ]
    for mod in required:
        try:
            __import__(mod)
            r.ok(f"import {mod}")
        except ImportError as exc:
            r.fail(f"import {mod}", str(exc))

    # PaddleOCR: the actual OCR engine (see pipeline/ocr.py, ocr_gpu.py).
    # Import failure here usually means paddlepaddle wasn't installed
    # separately first — see requirements.txt for the exact command.
    try:
        import paddleocr  # noqa: F401
        r.ok("import paddleocr")
    except ImportError as exc:
        r.fail(
            "import paddleocr",
            f"{exc} — install paddlepaddle first (CPU: pip install "
            f"paddlepaddle==3.0.0 / GPU: paddlepaddle-gpu==2.6.2), "
            f"then paddleocr>=2.9.0",
        )


# ── 2. Settings ─────────────────────────────────────────────────────────

def check_settings(r: Results) -> None:
    section("Settings / config")
    try:
        from aic26.config import settings
        r.ok("aic26.config.settings imports")
    except Exception as exc:
        r.fail("aic26.config.settings imports", str(exc))
        return

    try:
        db_path = str(settings.db_path)
        faiss_path = str(settings.faiss_index_path)
        if db_path == faiss_path:
            r.fail(
                "db_path != faiss_index_path",
                f"both resolve to {db_path!r} — VectorDatabase (query "
                f"history) and IndexBuilder (corpus index) will overwrite "
                f"each other. Check .env for a stale DB_PATH override.",
            )
        else:
            r.ok(f"db_path ({db_path}) != faiss_index_path ({faiss_path})")
    except AttributeError as exc:
        r.fail(
            "faiss_index_path exists on Settings",
            f"{exc} — the db_path split patch may not be applied",
        )

    for field_name in ("clip32_embedding_dir", "use_dual_embedding", "yolo_model_name"):
        if hasattr(settings, field_name):
            r.ok(f"settings.{field_name} = {getattr(settings, field_name)!r}")
        else:
            r.fail(f"settings.{field_name}", "field missing")


# ── 3. Filesystem / data paths ───────────────────────────────────────────

def check_paths(r: Results) -> None:
    section("Data directories (existence only — not content)")
    try:
        from aic26.config import settings
    except Exception as exc:
        r.skip("path checks", f"settings import failed: {exc}")
        return

    import os

    for field_name in (
        "keyframes_dir", "embedding_dir", "detection_dir",
        "metadata_dir", "map_keyframe_dir", "ocr_csv_path",
    ):
        path = str(getattr(settings, field_name, ""))
        if not path:
            r.skip(field_name, "not configured")
            continue
        if os.path.exists(path):
            r.ok(f"{field_name} exists ({path})")
        else:
            r.skip(field_name, f"does not exist yet ({path}) — fine before first pipeline run")


# ── 4. YOLO26 detector loads ─────────────────────────────────────────────

def check_yolo(r: Results, heavy: bool) -> None:
    section("YOLO26 detector")
    if not heavy:
        r.skip("YOLO26 model load", "--skip-heavy")
        return
    try:
        from aic26.config import settings
        from ultralytics import YOLO

        model_name = settings.yolo_model_name
        print(f"  Loading {model_name} (downloads on first use if missing)...")
        model = YOLO(model_name)
        r.ok(f"YOLO('{model_name}') loads")

        # Quick inference smoke test on a tiny synthetic image — confirms
        # the model actually runs, not just that the weights downloaded.
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model.predict(source=dummy, conf=0.1, verbose=False)
        r.ok(f"YOLO predict() runs (returned {len(results)} result(s))")
    except Exception as exc:
        r.fail("YOLO26 model load/predict", f"{type(exc).__name__}: {exc}")


# ── 5. Translation engine ────────────────────────────────────────────────

def check_translation(r: Results, heavy: bool) -> None:
    section("Vi->En translation")
    try:
        from aic26.core.translation import get_translation_engine, looks_vietnamese
        r.ok("aic26.core.translation imports")
    except Exception as exc:
        r.fail("aic26.core.translation imports", str(exc))
        return

    if not looks_vietnamese("một người đàn ông đang chạy"):
        r.fail("looks_vietnamese() heuristic", "failed to detect Vietnamese diacritics")
    else:
        r.ok("looks_vietnamese() detects Vietnamese text")

    if not heavy:
        r.skip("live translation call", "--skip-heavy (mode=test uses a local model download)")
        return

    try:
        engine = get_translation_engine()
        engine.set_mode("test")  # local model only, no API cost
        original = "một người đàn ông đang chạy trên bãi biển"
        translated = engine.translate(original)
        if translated.strip().lower() == original.strip().lower():
            r.fail("translate() actually translates", "output == input, translation may be a no-op")
        else:
            r.ok(f"translate() works: '{original}' -> '{translated}'")
    except Exception as exc:
        r.fail("translate() call", f"{type(exc).__name__}: {exc}")


# ── 6. SearchEngine wiring (no data required, just object construction) ──

def check_search_wiring(r: Results) -> None:
    section("SearchEngine <-> translator wiring (static check)")
    try:
        import inspect
        from aic26.core.search import SearchEngine

        src = inspect.getsource(SearchEngine)
        if "self.translator" in src and "get_translation_engine" in src:
            r.ok("SearchEngine references self.translator / get_translation_engine")
        else:
            r.fail(
                "SearchEngine translator wiring",
                "no reference to self.translator found in source — "
                "the translation patch may not be applied",
            )
        if "query_en" in src:
            r.ok("search_text/search_text_all use a translated query_en variable")
        else:
            r.fail("query_en usage", "search_text may still be encoding the raw query")
    except Exception as exc:
        r.fail("SearchEngine source inspection", f"{type(exc).__name__}: {exc}")


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-heavy", action="store_true",
        help="Skip checks that load real models (YOLO26 weights, translation model)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated subset: imports,settings,paths,yolo,translation,search",
    )
    args = parser.parse_args()

    selected = set(args.only.split(",")) if args.only else None
    heavy = not args.skip_heavy

    r = Results()
    checks = [
        ("imports", lambda: check_imports(r)),
        ("settings", lambda: check_settings(r)),
        ("paths", lambda: check_paths(r)),
        ("yolo", lambda: check_yolo(r, heavy)),
        ("translation", lambda: check_translation(r, heavy)),
        ("search", lambda: check_search_wiring(r)),
    ]

    print("=" * 70)
    print("AIC-26 smoke test")
    print("=" * 70)

    for name, fn in checks:
        if selected and name not in selected:
            continue
        try:
            fn()
        except Exception:  # noqa: BLE001 — a check crashing shouldn't stop the rest
            print(f"  💥 Check '{name}' crashed:")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Summary: {len(r.passed)} passed, {len(r.failed)} failed, {len(r.skipped)} skipped")
    if r.failed:
        print("\nFailed checks:")
        for name, reason in r.failed:
            print(f"  - {name}: {reason}")
    print("=" * 70)

    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())