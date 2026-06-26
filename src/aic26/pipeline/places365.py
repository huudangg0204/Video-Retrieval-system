"""
Places365 scene classifier.

Produces per-keyframe scene labels (top-K categories with confidence)
used by the metadata pipeline. Modeled on lifeXplore 2024 §3 Table 1,
which uses EfficientNet-B2 trained on Places365 (365 scene categories).

Why Places365 for scene?
------------------------
EVA-02 captures scene semantics indirectly via image-level embeddings,
but offers no explicit category labels. A dedicated Places365 head gives
interpretable labels like "kitchen", "beach", "skyscraper" that map
cleanly to user queries like "find all beach scenes". Combined with
the scene-graph triples, this is what enables the `-c kitchen` filter
(lifeXplore §3 Table 2 special commands) and `-r "person in kitchen"`
relational queries.

Design notes
------------
- Backend: torchvision EfficientNet-B2 with the final FC layer replaced
  by a 365-class head. Weights are NOT bundled with torchvision (the
  default ImageNet weights have 1000 classes) — they must be downloaded
  separately via Task T10 (`aic26 download places365`).
- For testability and to keep MVP cheap, this module accepts an
  injected `model` and `labels_path`. Production use constructs via
  `Places365Classifier.from_settings(settings)` which wires up the real
  weights path.
- Inference: model.eval() + torch.no_grad(); output is softmax over
  365 logits → top-K (category, confidence) pairs.

References
----------
- lifeXplore 2024 §3 Table 1: model = EfficientNet B2 Places365.
- docs/specs/metadata-extraction.md §Open Questions 4.
- Places365 dataset: http://places.csail.mit.edu/
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# Input image size for EfficientNet-B2 (224×224 is the standard ImageNet
# pretraining resolution; Places365 fine-tunes at the same resolution).
PLACES365_INPUT_SIZE: int = 224

# Number of scene categories in the Places365 dataset.
PLACES365_NUM_CATEGORIES: int = 365


@dataclass
class Places365Result:
    """Top-K scene classification for one image.

    Attributes:
        category: Top-1 scene label (e.g., "kitchen").
        confidence: Top-1 softmax probability in [0, 1].
        top5: List of (category, confidence) tuples sorted by confidence
            descending. Length matches the requested top_k (default 5).
    """

    category: str
    confidence: float
    top5: list[tuple[str, float]] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Places365Result(category={self.category!r}, confidence={self.confidence:.3f})"


def load_places365_labels(labels_file: Path) -> list[str]:
    """Parse the Places365 `categories_places365.txt` label file.

    The file format (MIT-CSAIL) is one line per category, where each
    line is a relative path like `/a/kitchen/kitchen`. We extract the
    leaf directory name as the human-readable label and lowercase it
    for consistent matching against user queries.

    Args:
        labels_file: Path to `categories_places365.txt`.

    Returns:
        List of 365 label strings, lowercase, indexed by category id.

    Raises:
        FileNotFoundError: If labels_file does not exist.
    """
    if not labels_file.exists():
        raise FileNotFoundError(f"Places365 labels file not found: {labels_file}")

    labels: list[str] = []
    for raw_line in labels_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Format: "/a/kitchen/kitchen" → leaf segment = "kitchen"
        leaf = line.rstrip("/").split("/")[-1]
        labels.append(leaf.lower())
    return labels


class Places365Classifier:
    """Scene classifier backed by a Places365-trained EfficientNet-B2.

    Args:
        model: A torch.nn.Module that accepts a [1, 3, 224, 224] tensor
            and returns logits of shape [1, 365]. Pass None to defer
            loading until first `classify()` call.
        labels_path: Path to `categories_places365.txt`. Labels are
            loaded lazily on first `classify()` call if not preloaded.
        device: Torch device string ('cuda', 'cpu', 'cuda:0', ...).
        top_k: Default number of categories returned per classify() call.
    """

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        labels_path: Path | None = None,
        device: str = "cpu",
        top_k: int = 5,
    ) -> None:
        self._model = model
        self.labels_path = labels_path
        self.device = device
        self.top_k = top_k
        self._labels: list[str] | None = None

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Use Places365Classifier.from_settings() "
                "to construct with weights, or pass model=... explicitly."
            )
        return self._model

    @property
    def labels(self) -> list[str]:
        if self._labels is None:
            if self.labels_path is None:
                raise RuntimeError("labels_path is required to load Places365 labels.")
            self._labels = load_places365_labels(self.labels_path)
        return self._labels

    def classify(self, image: Image.Image, top_k: int | None = None) -> Places365Result:
        """Classify one image into top-K Places365 scene categories.

        Args:
            image: PIL Image, any size. Will be resized to 224×224.
            top_k: Number of categories to return. Defaults to self.top_k.

        Returns:
            Places365Result with top-1 category + confidence and top5 list.

        Raises:
            ValueError: If image mode is not RGB / L.
            RuntimeError: If labels are missing for the top index.
        """
        if top_k is None:
            top_k = self.top_k
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        # 1. Preprocess: resize → tensor → normalize (ImageNet stats).
        if image.mode != "RGB":
            image = image.convert("RGB")
        resized = image.resize((PLACES365_INPUT_SIZE, PLACES365_INPUT_SIZE))
        arr_np = np.ascontiguousarray(_pil_to_numpy(resized))
        arr = torch.from_numpy(arr_np).permute(2, 0, 1).unsqueeze(0)
        arr = arr.to(self.device).float() / 255.0
        # ImageNet normalization (Places365 uses the same convention).
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        normalized = (arr - mean) / std

        # 2. Forward pass under no_grad.
        model = self.model
        model.eval()
        with torch.no_grad():
            logits = model(normalized)

        # 3. Softmax → top-K.
        probs = torch.softmax(logits, dim=1)[0]
        k = min(top_k, probs.shape[0])
        topk = torch.topk(probs, k=k)
        indices = topk.indices.cpu().tolist()
        confidences = topk.values.cpu().tolist()

        labels = self.labels
        top_pairs: list[tuple[str, float]] = []
        for idx, conf in zip(indices, confidences):
            if idx >= len(labels):
                raise RuntimeError(
                    f"Model output index {idx} out of range for "
                    f"{len(labels)} Places365 labels."
                )
            top_pairs.append((labels[idx], float(conf)))

        top1_label, top1_conf = top_pairs[0]
        return Places365Result(
            category=top1_label,
            confidence=float(top1_conf),
            top5=top_pairs,
        )


def _pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a writable numpy array (H, W, 3) uint8.

    PIL's __array_interface__ returns a read-only view; we copy so that
    torch.from_numpy does not warn about non-writable tensors.
    """
    return np.array(image, dtype="uint8")
