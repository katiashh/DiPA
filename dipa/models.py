from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from .backbones import insightface, magface
from .io import sha256
from .patch import resize


class FaceEncoder(nn.Module):
    def __init__(self, backbone, name, color_order="rgb", mean=(0.5,) * 3,
                 std=(0.5,) * 3, input_size=(112, 112)):
        super().__init__()
        if color_order not in {"rgb", "bgr"}:
            raise ValueError("color_order must be rgb or bgr")
        if len(mean) != 3 or len(std) != 3 or min(std) <= 0:
            raise ValueError("mean/std must contain three values; std must be positive")
        self.backbone = backbone.eval().requires_grad_(False)
        self.name = name
        self.color_order = color_order
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, image):
        image = resize(image, self.input_size)
        if self.color_order == "bgr":
            image = image[:, [2, 1, 0]]
        features = self.backbone((image - self.mean) / self.std)
        if features.ndim != 2 or features.shape[0] != image.shape[0]:
            raise ValueError(f"{self.name} must return [batch, embedding_dim]")
        return features


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a state_dict or a checkpoint containing state_dict")
    for key in ("state_dict", "model_state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("Checkpoint state_dict is empty or invalid")
    return checkpoint


def map_state_dict(state, expected, prefix=None):
    if prefix is None:
        prefixes = ("", "module.", "module.features.", "features.module.",
                    "features.", "backbone.", "module.backbone.")
    else:
        prefixes = (prefix,)
    for candidate in prefixes:
        mapped = {key[len(candidate):]: value for key, value in state.items()
                  if key.startswith(candidate)}
        if set(expected).issubset(mapped):
            wrong = [key for key in expected
                     if not isinstance(mapped[key], torch.Tensor)
                     or mapped[key].shape != expected[key].shape]
            if not wrong:
                ignored = sorted(key for key in mapped if key not in expected)
                return {key: mapped[key] for key in expected}, candidate, ignored
    raise ValueError("Checkpoint does not fully match this backbone. Check backend, "
                     "architecture, embedding_size and state_prefix; no partial loading is allowed.")


def load_models(specs, device, require_ensemble=True):
    names = [spec["name"] for spec in specs]
    if not specs or len(set(names)) != len(names):
        raise ValueError("Model names must be nonempty and unique")
    if require_ensemble and set(names) != {"arcface", "cosface", "magface"}:
        raise ValueError("Generation requires arcface, cosface and magface")
    paths = [Path(spec["checkpoint"]).expanduser().resolve() for spec in specs]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}. See weights/README.md")
    hashes = [sha256(path) for path in paths]
    if require_ensemble and len(set(hashes)) != len(hashes):
        raise ValueError("The ensemble requires three different trained checkpoints")
    models, metadata = [], []
    for spec, path, digest in zip(specs, paths, hashes):
        backend = spec["backend"]
        modules = {"insightface": insightface, "magface": magface}
        if backend not in modules:
            raise ValueError(f"Unknown backend: {backend}")
        architecture = spec["architecture"]
        if architecture not in modules[backend].__all__:
            raise ValueError(f"Unknown {backend} architecture: {architecture}")
        kwargs = {"num_features" if backend == "insightface" else "num_classes":
                  spec.get("embedding_size", 512)}
        backbone = getattr(modules[backend], architecture)(**kwargs)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state, prefix, ignored = map_state_dict(
            extract_state_dict(checkpoint), backbone.state_dict(), spec.get("state_prefix"))
        backbone.load_state_dict(state, strict=True)
        encoder = FaceEncoder(backbone, spec["name"], spec["color_order"],
                              spec["mean"], spec["std"], spec.get("input_size", [112, 112]))
        models.append(encoder.to(device).eval())
        metadata.append({**spec, "checkpoint": str(path), "sha256": digest,
                         "loaded_prefix": prefix, "ignored_keys": ignored})
        del checkpoint, state
    return models, metadata
