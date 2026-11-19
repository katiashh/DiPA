from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .patch import apply_patch, render_patch, total_variation


@dataclass
class AttackResult:
    latent_patch: torch.Tensor
    history: list
    best_step: int


def validate_attack(config, canvas_size):
    required = {"steps", "learning_rate", "patch_size", "screen_size", "median_kernel",
                "box", "tv_weight", "seed", "log_every"}
    if set(config) != required:
        raise ValueError(f"Attack keys mismatch: missing {required - set(config)}, "
                         f"unknown {set(config) - required}")
    for key in ("steps", "patch_size", "median_kernel", "log_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if config["median_kernel"] % 2 != 1 or config["median_kernel"] // 2 >= config["patch_size"]:
        raise ValueError("Invalid median_kernel")
    if not 0 < config["learning_rate"] < float("inf") or not 0 <= config["tv_weight"] < float("inf"):
        raise ValueError("learning_rate must be positive and tv_weight nonnegative")
    for key, size in (("screen_size", config["screen_size"]), ("canvas_size", canvas_size)):
        if len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
            raise ValueError(f"{key} must be [height, width] in positive integers")
    box = config["box"]
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("box must be [x, y, width, height] in integers")
    x, y, w, h = box
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > canvas_size[1] or y + h > canvas_size[0]:
        raise ValueError("box must lie entirely inside canvas_size")
    sh, sw = config["screen_size"]
    if abs(sw / sh - w / h) > 0.01:
        raise ValueError("screen_size and box must have the same aspect ratio")


@torch.no_grad()
def cosine_scores(models, image, reference):
    scores = {}
    for model in models:
        value = F.cosine_similarity(model(image), model(reference)).mean()
        if not torch.isfinite(value):
            raise RuntimeError(f"Non-finite cosine similarity from {model.name}")
        scores[model.name] = float(value)
    return scores


def optimize(models, image, config, callback=None):
    validate_attack(config, image.shape[-2:])
    device = image.device
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    side = config["patch_size"]
    patch = torch.rand((1, 3, side, side), device=device, generator=generator).requires_grad_()
    optimizer = torch.optim.Adam([patch], lr=config["learning_rate"])
    with torch.no_grad():
        reference = [model(image).detach() for model in models]
    best_loss, best_step, best_patch = float("inf"), 0, None
    history = []
    for step in range(config["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = {}
        for model, clean_features in zip(models, reference):
            screen = render_patch(patch, config["median_kernel"], config["screen_size"])
            patched = apply_patch(image, screen, config["box"])
            score = F.cosine_similarity(model(patched), clean_features).mean()
            if not torch.isfinite(score):
                raise RuntimeError(f"Non-finite loss at step {step}: {model.name}")
            if step < config["steps"]:
                score.backward()
            scores[model.name] = float(score.detach())
        tv = total_variation(patch)
        if config["tv_weight"] and step < config["steps"]:
            (config["tv_weight"] * tv).backward()
        loss = sum(scores.values()) + config["tv_weight"] * float(tv.detach())
        row = {"step": step, "loss": loss, "cosine_sum": sum(scores.values()),
               "tv": float(tv.detach()), **scores}
        history.append(row)
        if loss < best_loss:
            best_loss, best_step, best_patch = loss, step, patch.detach().clone()
        if callback is not None and (step % config["log_every"] == 0 or step == config["steps"]):
            callback(row)
        if step < config["steps"]:
            if patch.grad is None or not torch.isfinite(patch.grad).all():
                raise RuntimeError(f"Invalid patch gradient at step {step}")
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0, 1)
    return AttackResult(best_patch, history, best_step)
