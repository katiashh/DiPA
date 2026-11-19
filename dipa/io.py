import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .patch import resize


def load_image(path, size=None):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    tensor = tensor.float().unsqueeze(0) / 255
    return resize(tensor, size) if size is not None else tensor


def save_image(tensor, path):
    image = tensor.detach().cpu().squeeze(0).clamp(0, 1)
    array = image.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)


def save_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
