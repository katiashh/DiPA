import argparse
import json
from pathlib import Path

import torch

from dipa.attack import cosine_scores
from dipa.io import load_image, save_json, sha256
from dipa.models import load_models


def main():
    parser = argparse.ArgumentParser(description="Compare clean and patched images using local FR models.")
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--patched", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/dipa.json"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output file already exists")
    config = json.loads(args.config.read_text())
    models, metadata = load_models(config["models"], args.device, require_ensemble=False)
    clean = load_image(args.clean, config["canvas_size"]).to(args.device)
    patched = load_image(args.patched, config["canvas_size"]).to(args.device)
    scores = cosine_scores(models, patched, clean)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json({"cosine_similarity": scores, "models": metadata,
               "clean_sha256": sha256(args.clean), "patched_sha256": sha256(args.patched),
               "canvas_size": config["canvas_size"]}, args.output)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
