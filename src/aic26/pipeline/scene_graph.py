"""
Scene-graph extraction via a vision-language MLLM.

Produces per-keyframe 3-tuple scene graphs `(subject, predicate, object)`
following the SnapSeek 3.0 §3.5 representation (LSC'25 paper). Enables
relational queries like `person hold cup` that pure image-level
embeddings (EVA-02) cannot capture, because the predicate describes
an interaction between two entities.

Why MLLM for scene graph?
-------------------------
EVA-02 captures holistic image-text similarity but loses object-object
relationships. Generating a structured scene graph via Qwen2.5-VL-7B
gives us discrete, indexable triples. The trade-off: each frame needs a
forward pass through a 7B-parameter model (~2-5s on RTX 2080 Ti 4-bit
quantized), which is why this step is opt-in via Settings.

Design notes
------------
- Pure functions: model and prompt are dependencies injected by the
  caller. Production code wires up the real Qwen model in T6
  (orchestrator). Tests use mocks — no GPU / network required.
- Robust JSON parsing: model output may include markdown fences,
  surrounding prose, or trailing commas. We extract the first JSON
  object from the response and parse it.
- Normalization: lowercase subjects/objects, strip whitespace,
  single-word nouns joined by underscore for clean query matching.
- Retry with temperature=0 on parse failure (deterministic regeneration).
  After `max_retries` exhausted, return [] (graceful degradation,
  log warning) — never crash the metadata pipeline over one bad frame.

References
----------
- SnapSeek 3.0 §3.5, LSC'25: 3-tuple (subject, predicate, object),
  Qwen2.5-VL-7B with carefully-crafted prompt enforcing lowercase
  noun, base-form verb, no passive.
- docs/specs/metadata-extraction.md §Open Questions 5.
"""

from __future__ import annotations

import json
import logging
import re

from PIL import Image

from aic26.config.constants import (
    SCENE_GRAPH_MAX_NEW_TOKENS,
    SCENE_GRAPH_MAX_TRIPLES,
    SCENE_GRAPH_PROMPT,
)

logger = logging.getLogger(__name__)


# ── Prompt builder ─────────────────────────────────────────────────────


def build_scene_graph_prompt() -> str:
    """Return the SnapSeek §3.5 scene-graph extraction prompt.

    The prompt instructs the model to:
    - emit lowercase noun subjects/objects
    - use base-form verb predicates
    - avoid passive constructions
    - output JSON only (no commentary), max 5 triples.
    """
    return SCENE_GRAPH_PROMPT


# ── Normalization ──────────────────────────────────────────────────────


def _norm_atom(atom, collapse_spaces: bool = True) -> str:
    """Normalize a single triple element.

    Args:
        atom: String or list/tuple of strings.
        collapse_spaces: If True, collapse internal whitespace to underscores
            (use for nouns: "red car" → "red_car"). If False, keep spaces
            (use for predicates: "next to" → "next to").
    """
    if isinstance(atom, (list, tuple)):
        # Model may emit e.g. ["red car"] instead of "red car".
        atom = " ".join(str(a) for a in atom)
    s = str(atom).strip().lower()
    if collapse_spaces:
        s = re.sub(r"\s+", "_", s)
    return s


def normalize_triple(triple) -> tuple[str, str, str]:
    """Normalize a 3-element triple to (subject, predicate, object).

    Rules:
    - Subject and object: lowercased, whitespace-stripped, internal
      spaces collapsed to underscores ("red car" → "red_car") for clean
      query matching.
    - Predicate: lowercased and whitespace-stripped but spaces are kept
      ("next to" → "next to"), so multi-word predicates read naturally.
    - Empty strings raise ValueError.

    Args:
        triple: An iterable of length 3 (list, tuple, etc.).

    Returns:
        Tuple of (subject, predicate, object).

    Raises:
        ValueError: If triple does not have length 3 or any element is empty.
    """
    if len(triple) != 3:
        raise ValueError(f"Triple must have exactly 3 elements, got {len(triple)}")
    s, p, o = triple
    s_norm = _norm_atom(s, collapse_spaces=True)
    p_norm = _norm_atom(p, collapse_spaces=False)
    o_norm = _norm_atom(o, collapse_spaces=True)
    if not s_norm or not p_norm or not o_norm:
        raise ValueError(
            f"Triple has empty element after normalization: ({s_norm!r}, {p_norm!r}, {o_norm!r})"
        )
    return (s_norm, p_norm, o_norm)


# ── Response parser ────────────────────────────────────────────────────


def _extract_first_json_object(text: str) -> str | None:
    """Find the first top-level {...} JSON object containing a `triples` key.

    Uses bracket-depth tracking so the value may contain nested arrays
    and objects (e.g. `{"triples": [[...], [...]]}`). Stops at the first
    matching object.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    if '"triples"' in candidate or "'triples'" in candidate:
                        return candidate
                    break  # outer object closed; move to next "{"
        start = text.find("{", start + 1)
    return None


def parse_scene_graph_response(
    text: str,
    max_triples: int = SCENE_GRAPH_MAX_TRIPLES,
) -> list[tuple[str, str, str]]:
    """Parse an MLLM response into a list of normalized 3-tuples.

    Tolerant of common model output variants:
    - Wrapped in markdown fences (```json ... ```)
    - Surrounding prose before/after the JSON object
    - Whitespace and case variations

    Args:
        text: Raw model output string.
        max_triples: Maximum number of triples to return (default 5).

    Returns:
        List of (subject, predicate, object) tuples, deduplicated,
        lowercased, max `max_triples` long.

    Raises:
        ValueError: If no JSON object with a "triples" key is found, or
            the JSON is malformed, or the triples list is malformed.
    """
    if not text or not text.strip():
        raise ValueError("Empty model response")

    candidate = _extract_first_json_object(text)
    if candidate is None:
        raise ValueError(f"No JSON object with 'triples' key found in response: {text!r}")

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in model response: {exc}. Text: {text!r}") from exc

    if not isinstance(data, dict) or "triples" not in data:
        raise ValueError(f"JSON missing 'triples' key: {data!r}")

    raw_triples = data["triples"]
    if not isinstance(raw_triples, list):
        raise ValueError(f"'triples' is not a list: {type(raw_triples).__name__}")

    parsed: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_triples:
        try:
            triple = normalize_triple(raw)
        except ValueError:
            # Skip malformed triple (wrong length, empty atom) — keep going.
            logger.debug("Skipping malformed triple: %r", raw)
            continue
        if triple in seen:
            continue  # Dedup.
        seen.add(triple)
        parsed.append(triple)
        if len(parsed) >= max_triples:
            break
    return parsed


# ── End-to-end extractor (with retry) ─────────────────────────────────


def extract_scene_graph(
    image: Image.Image,
    model,
    processor,
    prompt: str | None = None,
    max_triples: int = SCENE_GRAPH_MAX_TRIPLES,
    max_new_tokens: int = SCENE_GRAPH_MAX_NEW_TOKENS,
    max_retries: int = 2,
) -> list[tuple[str, str, str]]:
    """Run scene-graph extraction on one image, with retry on parse failure.

    Args:
        image: PIL Image (any size).
        model: A `Qwen2_5_VLForConditionalGeneration` (or compatible)
            instance loaded via `.from_pretrained()`. HF Transformers
            models expose `.generate()`, not a `.chat()` method — this
            function drives inference through `processor` +
            `model.generate()` directly.
        processor: The matching `AutoProcessor.from_pretrained(...)` for
            `model`. Builds the chat-templated prompt and turns
            (image, text) into model inputs.
        prompt: Custom prompt. Defaults to SCENE_GRAPH_PROMPT.
        max_triples: Maximum triples to return.
        max_new_tokens: Max tokens for the model's response.
        max_retries: Number of attempts before giving up.

    Returns:
        List of (subject, predicate, object) tuples, max `max_triples` long.
        Returns [] if all attempts fail (graceful degradation).
    """
    if prompt is None:
        prompt = build_scene_graph_prompt()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            import torch

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # deterministic, mirrors old temperature=0
                )
            # Strip the prompt tokens — generate() returns prompt + completion.
            trimmed_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            triples = parse_scene_graph_response(
                str(response),
                max_triples=max_triples,
            )
            if attempt > 1:
                logger.info(
                    "Scene graph parsed successfully on attempt %d/%d",
                    attempt,
                    max_retries,
                )
            return triples
        except ValueError as exc:
            last_error = str(exc)
            logger.warning(
                "Scene graph parse failed (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
            )

    # All attempts exhausted — graceful degradation.
    logger.error(
        "Scene graph extraction failed after %d attempts. Last error: %s",
        max_retries,
        last_error,
    )
    return []