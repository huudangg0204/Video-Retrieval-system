"""
Tests for scripts/download_places365.py.

Avoids actually hitting the network. Stubs `urllib.request.urlopen`
with a context manager that returns a fake response.

Covers:
1. Both files download from default URL when missing.
2. Skip re-download when file already present (no flag).
3. --force re-downloads even when present.
4. --weights-only / --labels-only filters.
5. Custom --url-prefix is reflected in the GET.
6. Network error → exit code 1 + cleans up partial file.
7. parse_args defaults + flags.
"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download_places365 as p365  # type: ignore[import-not-found]  # noqa: E402

# ── Fake response ─────────────────────────────────────────────────────


class _FakeResponse:
    """Minimal urllib-like response. set_content_size sets Content-Length header."""

    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._buf.close()

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def close(self) -> None:
        pass


@contextmanager
def fake_urlopen(response: object):
    """Patch urllib.request.urlopen to return `response`."""
    import urllib.request

    orig = urllib.request.urlopen
    urllib.request.urlopen = lambda url, timeout=None: response  # type: ignore[assignment]
    try:
        yield
    finally:
        urllib.request.urlopen = orig


# ── parse_args ────────────────────────────────────────────────────────


class TestParseArgs:
    def test_defaults(self):
        args = p365.parse_args([])
        assert args.output == "models/places365"
        assert args.weights_url == p365.DEFAULT_WEIGHTS_URL
        assert args.labels_url == p365.DEFAULT_LABELS_URL
        assert args.weights_only is False
        assert args.labels_only is False
        assert args.force is False

    def test_overrides(self):
        args = p365.parse_args(
            [
                "--output",
                "/tmp/p365",
                "--weights-url",
                "https://example.com/w.pth",
                "--labels-url",
                "https://example.com/l.txt",
                "--weights-only",
                "--force",
            ]
        )
        assert args.output == "/tmp/p365"
        assert args.weights_url == "https://example.com/w.pth"
        assert args.labels_url == "https://example.com/l.txt"
        assert args.weights_only is True
        assert args.force is True


# ── main() end-to-end (with fake urlopen) ─────────────────────────────


class TestMain:
    def test_downloads_both_when_missing(self, tmp_path: Path):
        weights_payload = b"WEIGHTS_PAYLOAD" + b"\0" * 16
        labels_payload = b"LABELS_PAYLOAD\nanother line\n"

        calls: list[str] = []

        def fake_resp_factory(url, timeout=None):
            calls.append(url)
            if url.endswith(p365.WEIGHTS_FILENAME):
                return _FakeResponse(weights_payload)
            if url.endswith(p365.LABELS_FILENAME):
                return _FakeResponse(labels_payload)
            raise AssertionError(f"unexpected URL: {url}")

        import urllib.request

        orig = urllib.request.urlopen
        urllib.request.urlopen = fake_resp_factory  # type: ignore[assignment]
        try:
            rc = p365.main(["--output", str(tmp_path)])
        finally:
            urllib.request.urlopen = orig

        assert rc == 0
        assert len(calls) == 2
        assert (tmp_path / p365.WEIGHTS_FILENAME).exists()
        assert (tmp_path / p365.LABELS_FILENAME).exists()
        assert (tmp_path / p365.WEIGHTS_FILENAME).read_bytes() == weights_payload
        assert (tmp_path / p365.LABELS_FILENAME).read_bytes() == labels_payload

    def test_skips_existing_files(self, tmp_path: Path):
        # Pre-populate
        (tmp_path / p365.LABELS_FILENAME).write_text("a/b\nc/d\n")
        (tmp_path / p365.WEIGHTS_FILENAME).write_bytes(b"x" * 100)

        calls: list[str] = []

        def must_not_call(url, timeout=None):
            calls.append(url)
            raise AssertionError("should not have downloaded")

        import urllib.request

        orig = urllib.request.urlopen
        urllib.request.urlopen = must_not_call  # type: ignore[assignment]
        try:
            rc = p365.main(["--output", str(tmp_path)])
        finally:
            urllib.request.urlopen = orig

        assert rc == 0
        assert calls == []

    def test_force_redownloads_existing_files(self, tmp_path: Path):
        (tmp_path / p365.LABELS_FILENAME).write_text("placeholder")

        seen_urls: list[str] = []

        def resp_factory(url, timeout=None):
            seen_urls.append(url)
            if url.endswith(p365.WEIGHTS_FILENAME):
                return _FakeResponse(b"\0" * 16)
            if url.endswith(p365.LABELS_FILENAME):
                return _FakeResponse(b"a/b\n")
            raise AssertionError(url)

        import urllib.request

        orig = urllib.request.urlopen
        urllib.request.urlopen = resp_factory  # type: ignore[assignment]
        try:
            rc = p365.main(["--output", str(tmp_path), "--force"])
        finally:
            urllib.request.urlopen = orig

        assert rc == 0
        assert len(seen_urls) == 2  # both re-downloaded
        # Labels file now reflects downloaded content
        assert (tmp_path / p365.LABELS_FILENAME).read_text() == "a/b\n"

    def test_weights_only(self, tmp_path: Path):
        seen_urls: list[str] = []

        def resp_factory(url, timeout=None):
            seen_urls.append(url)
            return _FakeResponse(b"\0" * 16)

        import urllib.request

        orig = urllib.request.urlopen
        urllib.request.urlopen = resp_factory  # type: ignore[assignment]
        try:
            rc = p365.main(["--output", str(tmp_path), "--weights-only"])
        finally:
            urllib.request.urlopen = orig

        assert rc == 0
        assert len(seen_urls) == 1
        assert seen_urls[0].endswith(p365.WEIGHTS_FILENAME)
        assert not (tmp_path / p365.LABELS_FILENAME).exists()

    def test_labels_only(self, tmp_path: Path):
        seen_urls: list[str] = []

        def resp_factory(url, timeout=None):
            seen_urls.append(url)
            return _FakeResponse(b"a/b\nc/d\n")

        import urllib.request

        orig = urllib.request.urlopen
        urllib.request.urlopen = resp_factory  # type: ignore[assignment]
        try:
            rc = p365.main(["--output", str(tmp_path), "--labels-only"])
        finally:
            urllib.request.urlopen = orig

        assert rc == 0
        assert len(seen_urls) == 1
        assert seen_urls[0].endswith(p365.LABELS_FILENAME)
        assert not (tmp_path / p365.WEIGHTS_FILENAME).exists()

    def test_custom_labels_url(self, tmp_path: Path):
        seen_urls: list[str] = []

        def resp_factory(url, timeout=None):
            seen_urls.append(url)
            return _FakeResponse(b"a/b\n")

        import urllib.request

        orig = urllib.request.urlopen
        urllib.request.urlopen = resp_factory  # type: ignore[assignment]
        try:
            rc = p365.main(
                [
                    "--output",
                    str(tmp_path),
                    "--labels-only",
                    "--labels-url",
                    "https://example.com/places365/labels.txt",
                ]
            )
        finally:
            urllib.request.urlopen = orig

        assert rc == 0
        assert seen_urls[0] == "https://example.com/places365/labels.txt"

    def test_network_error_exits_1_and_cleans_up(self, tmp_path: Path):
        def boom(url, timeout=None):
            raise URLError("simulated network failure")

        import urllib.request

        orig = urllib.request.urlopen
        urllib.request.urlopen = boom  # type: ignore[assignment]
        try:
            rc = p365.main(
                [
                    "--output",
                    str(tmp_path),
                    "--weights-only",
                ]
            )
        finally:
            urllib.request.urlopen = orig

        assert rc == 1
        # Partial file should be cleaned up
        assert not (tmp_path / p365.WEIGHTS_FILENAME).exists()
