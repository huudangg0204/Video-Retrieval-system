"""
Unit tests for pipeline/scene_graph.py.

Tests cover:
1. Prompt builder — produces the SnapSeek §3.5 rules template.
2. JSON parser — robust against model output variants (with/without
   markdown fences, prose around JSON, trailing commas).
3. Normalization — lowercases, deduplicates, slices to top-K.
4. extract_scene_graph() — mocked model, retries on bad output, returns
   list of (subject, predicate, object) tuples.

The real Qwen2.5-VL-7B model is NOT loaded in tests; we inject a mock
that returns configurable strings. This keeps unit tests fast and
deterministic without GPU / internet.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from PIL import Image

from aic26.pipeline.scene_graph import (
    build_scene_graph_prompt,
    extract_scene_graph,
    normalize_triple,
    parse_scene_graph_response,
)

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def dummy_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


def _mock_model(return_value: str, call_log: list | None = None):
    """Build a mock model whose .chat() returns the given string.

    `model.chat` is a MagicMock so tests can introspect call_args.
    """
    model = MagicMock()
    chat_mock = MagicMock(return_value=return_value)
    if call_log is not None:
        chat_mock.side_effect = lambda *a, **kw: call_log.append("called") or return_value
    model.chat = chat_mock
    return model


# ── Prompt builder ─────────────────────────────────────────────────────


class TestBuildSceneGraphPrompt:
    def test_returns_non_empty_string(self):
        prompt = build_scene_graph_prompt()
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 50

    def test_includes_output_format(self):
        prompt = build_scene_graph_prompt()
        assert "JSON" in prompt
        assert "triples" in prompt.lower()

    def test_includes_snapseek_rules(self):
        prompt = build_scene_graph_prompt()
        # SnapSeek §3.5 rules
        assert "lowercase" in prompt.lower()
        assert "base" in prompt.lower()  # base-form verbs
        assert "passive" in prompt.lower()


# ── Triple normalization ───────────────────────────────────────────────


class TestNormalizeTriple:
    def test_lowercase(self):
        assert normalize_triple(("Person", "Hold", "Cup")) == ("person", "hold", "cup")

    def test_strip_whitespace(self):
        assert normalize_triple(("  person  ", " hold ", " cup ")) == ("person", "hold", "cup")

    def test_replace_internal_spaces_with_underscore_in_nouns(self):
        # SnapSeek uses single-word lowercase nouns; multi-word objects
        # should use underscore separator for query matching.
        assert normalize_triple(("person", "next to", ("red car",))) == (
            "person",
            "next to",
            "red_car",
        )

    def test_handles_string_object(self):
        # Acceptable if model returns flat string instead of tuple.
        assert normalize_triple(("person", "hold", "cup")) == ("person", "hold", "cup")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            normalize_triple(("", "hold", "cup"))
        with pytest.raises(ValueError):
            normalize_triple(("person", "", "cup"))


# ── Response parser ────────────────────────────────────────────────────


class TestParseSceneGraphResponse:
    def test_parses_clean_json(self):
        text = '{"triples": [["person", "hold", "cup"], ["cup", "on", "table"]]}'
        triples = parse_scene_graph_response(text)
        assert triples == [("person", "hold", "cup"), ("cup", "on", "table")]

    def test_parses_json_with_markdown_fence(self):
        text = '```json\n{"triples": [["person", "hold", "cup"]]}\n```'
        triples = parse_scene_graph_response(text)
        assert triples == [("person", "hold", "cup")]

    def test_parses_json_embedded_in_prose(self):
        text = (
            'Sure, here are the triples:\n{"triples": [["dog", "run", "park"]]}\nHope that helps!'
        )
        triples = parse_scene_graph_response(text)
        assert triples == [("dog", "run", "park")]

    def test_lowercases_three_tuples(self):
        text = '{"triples": [["PERSON", "HOLD", "CUP"]]}'
        triples = parse_scene_graph_response(text)
        assert triples == [("person", "hold", "cup")]

    def test_normalizes_whitespace(self):
        text = '{"triples": [["  Person  ", "  hold  ", " Cup "]]}'
        triples = parse_scene_graph_response(text)
        assert triples == [("person", "hold", "cup")]

    def test_slices_to_max_triples(self):
        # Distinct triples — dedup would otherwise collapse them to one.
        text = json.dumps({"triples": [[f"s{i}", "p", f"o{i}"] for i in range(10)]})
        triples = parse_scene_graph_response(text, max_triples=3)
        assert len(triples) == 3

    def test_dedups_identical_triples(self):
        text = '{"triples": [["person", "hold", "cup"], ["person", "hold", "cup"]]}'
        triples = parse_scene_graph_response(text)
        assert triples == [("person", "hold", "cup")]

    def test_empty_triples_returns_empty_list(self):
        text = '{"triples": []}'
        triples = parse_scene_graph_response(text)
        assert triples == []

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="[Jj]son"):
            parse_scene_graph_response("not json at all")

    def test_json_without_triples_key_raises(self):
        text = '{"results": []}'
        with pytest.raises(ValueError, match="triples"):
            parse_scene_graph_response(text)

    def test_triple_wrong_length_skipped(self):
        text = '{"triples": [["person", "hold"], ["cup", "on", "table"]]}'
        triples = parse_scene_graph_response(text)
        # Skipped: ["person", "hold"] is length 2; kept: ("cup", "on", "table")
        assert triples == [("cup", "on", "table")]


# ── extract_scene_graph end-to-end (mocked model) ──────────────────────


class TestExtractSceneGraph:
    def test_returns_parsed_triples(self, dummy_image):
        model = _mock_model('{"triples": [["person", "hold", "cup"], ["cup", "on", "table"]]}')
        triples = extract_scene_graph(dummy_image, model=model)
        assert triples == [("person", "hold", "cup"), ("cup", "on", "table")]

    def test_retries_on_bad_output_then_succeeds(self, dummy_image):
        call_log: list = []
        responses = iter(
            [
                "not json at all",  # first attempt fails
                '{"triples": [["person", "hold", "cup"]]}',  # second succeeds
            ]
        )

        chat_mock = MagicMock(
            side_effect=lambda *a, **kw: call_log.append("call") or next(responses)
        )
        model = MagicMock()
        model.chat = chat_mock
        triples = extract_scene_graph(dummy_image, model=model, max_retries=2)
        assert triples == [("person", "hold", "cup")]
        assert len(call_log) == 2  # model called twice

    def test_returns_empty_after_exhausted_retries(self, dummy_image):
        model = _mock_model("still not json")
        triples = extract_scene_graph(dummy_image, model=model, max_retries=2)
        # Graceful degradation: empty list, not exception.
        assert triples == []

    def test_respects_max_triples(self, dummy_image):
        text = json.dumps({"triples": [["s", "p", f"o{i}"] for i in range(10)]})
        model = _mock_model(text)
        triples = extract_scene_graph(dummy_image, model=model, max_triples=3)
        assert len(triples) == 3

    def test_passes_image_to_model(self, dummy_image):
        model = _mock_model('{"triples": []}')
        extract_scene_graph(dummy_image, model=model)
        # chat() was called positionally with messages as first arg.
        args, kwargs = model.chat.call_args
        messages = kwargs.get("messages") or (args[0] if args else None)
        assert messages is not None
        # First message contains content with image+text
        first_msg = messages[0]
        content = first_msg["content"]
        # Find the image part
        image_parts = [c for c in content if isinstance(c, dict) and c.get("type") == "image"]
        assert len(image_parts) >= 1

    def test_custom_prompt_used(self, dummy_image):
        custom = "Return ONLY JSON. NO prose."
        model = _mock_model('{"triples": []}')
        extract_scene_graph(dummy_image, model=model, prompt=custom)
        # Verify the prompt text was passed in messages
        args, kwargs = model.chat.call_args
        messages = kwargs.get("messages") or (args[0] if args else None)
        text_parts = [
            c["text"]
            for c in messages[0]["content"]
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        assert any(custom in t for t in text_parts)
