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
- Backbone: torchvision ResNet50 (default) or EfficientNet-B2 with the
  final FC layer replaced by a 365-class head. The official CSAIL
  Places365 release ships ResNet{18,50,152} / AlexNet / DenseNet161 /
  GoogLeNet / VGG16 checkpoints; we default to ResNet50 because it's
  the lightest canonical option (~98 MB). See `scripts/download_places365.py`.
- The checkpoint is a `.pth.tar` dict with keys {state_dict, epoch,
  arch, best_prec1} and `module.`-prefixed DataParallel weights. We
  strip the prefix and remap the FC head to 365 classes.
- For testability and to keep MVP cheap, this module accepts an
  injected `model` and `labels_path`. Production use constructs via
  `Places365Classifier.from_weights(weights_path, labels_path, ...)`,
  which rebuilds the architecture and loads the state dict with
  strict=False so FC mismatches are tolerated.
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

    The CSAIL format (also used by upstream Places365 release) is one
    line per category, where each line is a space-separated
    ``"<label> <numeric_id>"`` pair, e.g.::

        /a/airfield 0
        /a/airplane_cabin 1
        /a/airport_terminal 2

    We return the label *only* (no path, no id), lowercased, indexed
    by category id. This matches what downstream code expects when it
    looks up a category by index.

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
        # Format: "<category_path> <numeric_id>"  → take the first token
        # and discard the trailing id.
        first_token = line.split()[0]
        # Strip the leading "/a/" or "/y/" path prefix to expose the leaf.
        leaf = first_token.rsplit("/", 1)[-1]
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
                "Model not loaded. Use Places365Classifier.from_weights() "
                "to construct with weights, or pass model=... explicitly."
            )
        return self._model

    @classmethod
    def from_weights(
        cls,
        weights_path: Path | str,
        labels_path: Path | str,
        device: str = "cpu",
        top_k: int = 5,
    ) -> Places365Classifier:
        """Build a Places365 classifier from a downloaded `.pth.tar` file.

        This is the canonical constructor used by `smoke_test_metadata.py
        --real-models` and by any future production wiring.

        Steps:
          1. `torch.load(weights_path, map_location='cpu')`
          2. Pick backbone from the checkpoint's `arch` field
          3. Strip `module.` prefix (DataParallel artifacts)
          4. `load_state_dict(strict=False)` so FC head can be re-init

        Args:
            weights_path: Path to `resnet50_places365.pth.tar` (or similar).
            labels_path:  Path to `categories_places365.txt`.
            device:       Torch device string ('cuda', 'cpu', ...).
            top_k:        Default top-K for classify().

        Returns:
            Places365Classifier with model loaded and labels resolved lazily.

        Raises:
            FileNotFoundError: If weights_path doesn't exist.
            ValueError: If the checkpoint's arch is unsupported.
        """
        weights_p = Path(weights_path)
        labels_p = Path(labels_path)
        if not weights_p.exists():
            raise FileNotFoundError(f"Places365 weights not found: {weights_p}")

        state = torch.load(weights_p, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected Places365 checkpoint format: {type(state).__name__}")
        model = _build_model_from_state_dict(state)
        model.eval()
        model = model.to(device)
        return cls(model=model, labels_path=labels_p, device=device, top_k=top_k)

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
                    f"Model output index {idx} out of range for {len(labels)} Places365 labels."
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


# ── Model loader ──────────────────────────────────────────────────────


# Backbones we know how to build. Extend cautiously — each `arch` key
# must match the value CSAIL writes into the `.pth.tar` `arch` field.
SUPPORTED_ARCHITECTURES: frozenset[str] = frozenset({"resnet50"})


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove the `module.` prefix that DataParallel training leaves behind.

    Keys that don't carry the prefix are passed through untouched.
    """
    cleaned: dict = {}
    for key, value in state_dict.items():
        cleaned[key[len("module.") :] if key.startswith("module.") else key] = value
    return cleaned


def _build_resnet50_places365(num_classes: int = PLACES365_NUM_CATEGORIES) -> torch.nn.Module:
    """Construct a torchvision ResNet50 with a custom FC head.

    We do NOT call `pretrained=True` (that would download ImageNet
    weights on top of our Places365 checkpoint). We just want the
    architecture with the right output dim; the Places365 state dict
    is then loaded with `strict=False` so the final FC can be
    reinitialized if its shape doesn't match.
    """
    from torchvision.models import resnet50

    model = resnet50(weights=None, num_classes=num_classes)
    return model


def _build_model_from_state_dict(state_dict: dict) -> torch.nn.Module:
    """Pick a backbone based on the checkpoint's `arch` field and load weights.

    The FC layer is allowed to mismatch (strict=False) so checkpoints
    with a different number of classes still load — useful for
    fine-tunes that swap the head. Missing FC entries get a fresh
    random init for the requested 365-class output.
    """
    arch = state_dict.get("arch", "resnet50")
    if arch not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported Places365 backbone: {arch!r}. "
            f"Supported: {sorted(SUPPORTED_ARCHITECTURES)}"
        )
    if arch == "resnet50":
        model = _build_resnet50_places365()

    raw_sd = state_dict.get("state_dict", state_dict)
    cleaned_sd = _strip_module_prefix(raw_sd)
    # Drop FC head keys whose shape doesn't match the freshly-built model.
    # Matching-shape FC weights (the common case for 365-class checkpoints)
    # get loaded normally; mismatched ones (e.g. 100-class fine-tunes) get
    # reinitialized by `model.fc = nn.Linear(..., num_classes)`.
    fc_weight = cleaned_sd.get("fc.weight")
    if fc_weight is not None and hasattr(model, "fc"):
        try:
            target_shape = tuple(model.fc.weight.shape)
        except AttributeError:
            target_shape = None
        if target_shape is not None and tuple(fc_weight.shape) != target_shape:
            cleaned_sd = {k: v for k, v in cleaned_sd.items() if not k.startswith("fc.")}
            logger.warning(
                "Places365 FC head shape mismatch: ckpt=%s vs model=%s — reinitializing",
                tuple(fc_weight.shape),
                target_shape,
            )
    missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
    if unexpected:
        logger.warning("Places365 load: %d unexpected keys ignored", len(unexpected))
    if missing:
        logger.warning("Places365 load: %d missing keys (FC head reinit)", len(missing))
    return model
