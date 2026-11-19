import argparse
import csv
import json
import platform
from pathlib import Path

import torch

from dipa import __version__
from dipa.attack import cosine_scores, optimize, validate_attack
from dipa.io import load_image, save_image, save_json, seed_everything, sha256
from dipa.models import load_models
from dipa.patch import apply_patch, quantize, render_patch


def main():
    parser = argparse.ArgumentParser(description="Generate personalized DiPA patches.")
    parser.add_argument("--image", type=Path, required=True, help="Aligned face crop")
    parser.add_argument("--config", type=Path, default=Path("configs/dipa.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-patches", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["attack"]["seed"] = args.seed
    if args.steps is not None:
        config["attack"]["steps"] = args.steps
    validate_attack(config["attack"], config["canvas_size"])
    if args.num_patches < 1:
        parser.error("--num-patches must be positive")
    if config["attack"]["seed"] + args.num_patches > 2**32:
        parser.error("Patch seeds exceed the supported range")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory")
    image = load_image(args.image, config["canvas_size"]).to(args.device)
    models, metadata = load_models(config["models"], args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": __version__, "python": platform.python_version(),
                "torch": torch.__version__, "device": str(args.device),
                "input": str(args.image.resolve()), "input_sha256": sha256(args.image),
                "models": metadata, "config": config}
    save_json(manifest, args.output / "manifest.json")
    save_image(image, args.output / "clean.png")
    base_seed = config["attack"]["seed"]
    for index in range(args.num_patches):
        attack = {**config["attack"], "seed": base_seed + index}
        seed_everything(attack["seed"])
        output = args.output / f"patch_{index:03d}"
        output.mkdir()
        def report(row):
            print(f"patch={index} step={row['step']} loss={row['loss']:.6f} "
                  f"cosine_sum={row['cosine_sum']:.6f}", flush=True)
        result = optimize(models, image, attack, report)
        screen = render_patch(result.latent_patch, attack["median_kernel"], attack["screen_size"])
        saved_screen = quantize(screen)
        saved_image = quantize(apply_patch(image, saved_screen, attack["box"]))
        save_image(result.latent_patch, output / "latent_patch.png")
        save_image(saved_screen, output / "patch.png")
        save_image(saved_image, output / "patched.png")
        torch.save({"latent_patch": result.latent_patch.cpu(), "attack": attack,
                    "best_step": result.best_step}, output / "patch.pt")
        save_json({"attack": attack, "best_step": result.best_step,
                   "float_cosine": cosine_scores(models, apply_patch(image, screen, attack["box"]), image),
                   "saved_png_cosine": cosine_scores(models, saved_image, load_image(args.output / "clean.png").to(args.device))},
                  output / "metrics.json")
        with (output / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=result.history[0].keys())
            writer.writeheader()
            writer.writerows(result.history)
        print(f"Saved {output}; best step {result.best_step}", flush=True)


if __name__ == "__main__":
    main()
