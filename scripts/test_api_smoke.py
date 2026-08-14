"""
API smoke test — hits the REAL running Flask server over HTTP.

Unlike run_smoke_1video.py (which calls Python classes directly), this
exercises the actual network layer: routing, JSON (de)serialization,
CORS, error handlers, and the security/dispatch fixes applied earlier
(path traversal on /api/image, dual-stream dispatch on
/api/build_faiss_index) — none of which were ever tested through a real
HTTP request until now.

Prerequisites:
    1. Run start_server_smoke.ps1 in a separate terminal FIRST, and wait
       for "Server is now ready to accept requests."
    2. pip install requests   (already in requirements.txt)

Usage:
    python test_api_smoke.py
    python test_api_smoke.py --url http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import requests


@dataclass
class Results:
    passed: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    skipped: list = field(default_factory=list)

    def ok(self, name: str, extra: str = "") -> None:
        self.passed.append(name)
        print(f"  ✅ PASS  {name}" + (f" — {extra}" if extra else ""))

    def fail(self, name: str, reason: str) -> None:
        self.failed.append((name, reason))
        print(f"  ❌ FAIL  {name} — {reason}")

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append((name, reason))
        print(f"  ⏭️  SKIP  {name} — {reason}")


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="Base URL of the running server")
    parser.add_argument("--video", default="L21_V029", help="Video name present in the smoke dataset")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    r = Results()

    # ── 0. Server reachable at all? ──────────────────────────────────
    section("Server reachability")
    try:
        resp = requests.get(f"{base}/health", timeout=10)
        if resp.status_code == 200:
            r.ok("GET /health", resp.text.strip())
        else:
            r.fail("GET /health", f"status {resp.status_code}")
    except requests.exceptions.ConnectionError:
        r.fail("GET /health", "connection refused — is start_server_smoke.ps1 running?")
        print("\nCannot reach the server — stopping here. Start it first, then retry.")
        return 1
    except Exception as exc:
        r.fail("GET /health", str(exc))
        return 1

    # ── 1. Homepage (GUI shell) ──────────────────────────────────────
    section("Homepage / GUI shell")
    try:
        resp = requests.get(f"{base}/", timeout=15)
        if resp.status_code == 200 and "<html" in resp.text.lower():
            r.ok("GET / (index.html renders)", f"{len(resp.text)} bytes")
        else:
            r.fail("GET /", f"status {resp.status_code}, no <html> found")
    except Exception as exc:
        r.fail("GET /", str(exc))

    # ── 2. System stats ───────────────────────────────────────────────
    section("System stats")
    try:
        resp = requests.get(f"{base}/api/stats", timeout=15)
        data = resp.json()
        if resp.status_code == 200 and data.get("status") == "ready":
            r.ok("GET /api/stats", f"total_images={data.get('total_images')}")
        else:
            r.fail("GET /api/stats", f"status={resp.status_code} body={data}")
    except Exception as exc:
        r.fail("GET /api/stats", str(exc))

    # ── 3. Text search (core flow) ────────────────────────────────────
    section("Text search")
    search_hits = []
    try:
        resp = requests.post(
            f"{base}/api/search_text",
            json={"query": "một người đang đứng", "top_k": 3, "threshold": 0.0,
                  "required_objects": [], "object_mode": "any", "scene_group": ""},
            timeout=60,  # first call loads models if server didn't warm up yet
        )
        data = resp.json()
        results = data.get("results", data) if isinstance(data, dict) else data
        if resp.status_code == 200 and isinstance(results, list) and len(results) > 0:
            r.ok("POST /api/search_text", f"{len(results)} result(s)")
            search_hits = results
        else:
            r.fail("POST /api/search_text", f"status={resp.status_code} body={str(data)[:200]}")
    except Exception as exc:
        r.fail("POST /api/search_text", str(exc))

    # ── 4. OCR search (expect 0 results — smoke frames have no text — but no error) ─
    section("OCR search")
    try:
        resp = requests.post(
            f"{base}/api/search_ocr",
            json={"query": "a", "top_k": 3, "mode": "substring", "case_insensitive": True},
            timeout=15,
        )
        if resp.status_code == 200:
            r.ok("POST /api/search_ocr", "0 results expected (smoke frames have no on-screen text)")
        else:
            r.fail("POST /api/search_ocr", f"status={resp.status_code} body={resp.text[:200]}")
    except Exception as exc:
        r.fail("POST /api/search_ocr", str(exc))

    # ── 5. Browse / frame range ───────────────────────────────────────
    section("Browse & frame range")
    try:
        resp = requests.get(f"{base}/api/browse/folder/{args.video}", timeout=15)
        data = resp.json()
        if resp.status_code == 200 and data.get("count", 0) > 0:
            r.ok("GET /api/browse/folder/<video>", f"{data['count']} frame(s)")
        else:
            r.fail("GET /api/browse/folder/<video>", f"status={resp.status_code} body={str(data)[:200]}")
    except Exception as exc:
        r.fail("GET /api/browse/folder/<video>", str(exc))

    try:
        resp = requests.get(f"{base}/api/get_frames_range/{args.video}/3/2", timeout=15)
        if resp.status_code == 200:
            r.ok("GET /api/get_frames_range/...")
        else:
            r.fail("GET /api/get_frames_range/...", f"status={resp.status_code}")
    except Exception as exc:
        r.fail("GET /api/get_frames_range/...", str(exc))

    # ── 6. Image serving — the path traversal fix, tested for real ────
    section("Image serving (live security check)")
    try:
        resp = requests.get(f"{base}/api/image", params={"path": f"data/keyframes_smoke/{args.video}/001.jpg"}, timeout=15)
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image/"):
            r.ok("GET /api/image (valid path)", f"{len(resp.content)} bytes")
        else:
            r.fail("GET /api/image (valid path)", f"status={resp.status_code}")
    except Exception as exc:
        r.fail("GET /api/image (valid path)", str(exc))

    for attack, label in [
        ("../../../../../../Windows/win.ini", "relative traversal"),
        ("C:\\Windows\\win.ini", "absolute Windows path"),
    ]:
        try:
            resp = requests.get(f"{base}/api/image", params={"path": attack}, timeout=15)
            if resp.status_code == 403:
                r.ok(f"GET /api/image ({label}) blocked", "403 as expected")
            elif resp.status_code == 404:
                r.ok(f"GET /api/image ({label}) blocked", "404 (path didn't resolve — also acceptable)")
            else:
                r.fail(
                    f"GET /api/image ({label})",
                    f"status={resp.status_code} — expected 403! "
                    f"THIS WOULD BE A SECURITY REGRESSION if 200.",
                )
        except Exception as exc:
            r.fail(f"GET /api/image ({label})", str(exc))

    # ── 7. TRAKE ────────────────────────────────────────────────────
    section("TRAKE")
    try:
        resp = requests.post(
            f"{base}/api/trake",
            json={"events": ["a person standing"], "top_k": 5, "candidates": 5, "option": "Highest"},
            timeout=60,
        )
        data = resp.json()
        if resp.status_code == 200:
            r.ok("POST /api/trake", f"count={data.get('count')}, elapsed={data.get('elapsed')}s")
        else:
            r.fail("POST /api/trake", f"status={resp.status_code} body={str(data)[:200]}")
    except Exception as exc:
        r.fail("POST /api/trake", str(exc))

    # ── 8. Export (real submission-format check) ──────────────────────
    section("Export")
    export_filename = None
    if search_hits:
        try:
            resp = requests.post(
                f"{base}/api/export_results",
                json={"results": search_hits, "name": "smoke_test_export"},
                timeout=15,
            )
            data = resp.json()
            if resp.status_code == 200 and data.get("status") == "ok":
                export_filename = data.get("csv_path")
                r.ok("POST /api/export_results", f"-> {export_filename}")
            else:
                r.fail("POST /api/export_results", f"status={resp.status_code} body={str(data)[:200]}")
        except Exception as exc:
            r.fail("POST /api/export_results", str(exc))
    else:
        r.skip("POST /api/export_results", "no search results from step 3 to export")

    if export_filename:
        try:
            resp = requests.get(f"{base}/download/{export_filename}", timeout=15)
            if resp.status_code == 200 and len(resp.content) > 0:
                r.ok("GET /download/<csv>", f"{len(resp.content)} bytes")
            else:
                r.fail("GET /download/<csv>", f"status={resp.status_code}")
        except Exception as exc:
            r.fail("GET /download/<csv>", str(exc))

    # ── 9. Group by video ──────────────────────────────────────────────
    section("Group by video")
    if search_hits:
        try:
            resp = requests.post(
                f"{base}/api/group_by_video",
                json={"results": search_hits, "sort_by": "frame"},
                timeout=15,
            )
            if resp.status_code == 200:
                r.ok("POST /api/group_by_video")
            else:
                r.fail("POST /api/group_by_video", f"status={resp.status_code}")
        except Exception as exc:
            r.fail("POST /api/group_by_video", str(exc))
    else:
        r.skip("POST /api/group_by_video", "no search results from step 3 to group")

    # ── 10. Stored queries (should include the searches made above) ────
    section("Stored queries")
    try:
        resp = requests.get(f"{base}/api/stored_queries", timeout=15)
        data = resp.json()
        if resp.status_code == 200 and data.get("count", 0) > 0:
            r.ok("GET /api/stored_queries", f"{data['count']} stored")
        else:
            r.fail("GET /api/stored_queries", f"status={resp.status_code} body={str(data)[:200]}")
    except Exception as exc:
        r.fail("GET /api/stored_queries", str(exc))

    # ── 11. Rebuild FAISS index over HTTP — live re-test of the dual dispatch fix ──
    section("FAISS rebuild (live dual-stream dispatch check)")
    try:
        resp = requests.post(f"{base}/api/build_faiss_index", timeout=15)
        data = resp.json()
        if resp.status_code == 409:
            r.skip("POST /api/build_faiss_index", "a build was already in progress")
        elif resp.status_code == 200:
            r.ok("POST /api/build_faiss_index (started)")
            # Poll for completion
            for _ in range(30):
                time.sleep(1)
                status = requests.get(f"{base}/api/build_faiss_index/status", timeout=10).json()
                if status.get("status") in ("done", "error"):
                    break
            if status.get("status") == "done":
                built_dual = status.get("result", {}).get("dual", False)
                if built_dual:
                    r.ok("Rebuild used build_dual()", "dual-stream dispatch fix confirmed live")
                else:
                    r.fail(
                        "Rebuild used build_dual()",
                        "result.dual=False — rebuilt single-stream even though "
                        "clip32_embeddings_smoke has data. Check system.py fix.",
                    )
            else:
                r.fail("FAISS rebuild completion", f"final status={status}")
        else:
            r.fail("POST /api/build_faiss_index", f"status={resp.status_code} body={str(data)[:200]}")
    except Exception as exc:
        r.fail("POST /api/build_faiss_index", str(exc))

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Summary: {len(r.passed)} passed, {len(r.failed)} failed, {len(r.skipped)} skipped")
    if r.failed:
        print("\nFailed checks:")
        for name, reason in r.failed:
            print(f"  - {name}: {reason}")
    print("=" * 70)
    print(
        "\nNOT covered by this script (needs a real browser — see the manual "
        "GUI checklist): drag-and-drop image search, upload_image, actual "
        "click-through of the search results grid, TRAKE UI, export buttons."
    )

    return 1 if r.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())