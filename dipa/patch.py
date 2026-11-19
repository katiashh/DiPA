import torch
import torch.nn.functional as F


def median_pool(patch, kernel_size=3):
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("median_kernel must be a positive odd integer")
    pad = kernel_size // 2
    if pad >= min(patch.shape[-2:]):
        raise ValueError("median_kernel is too large for the patch")
    if pad:
        patch = F.pad(patch, (pad, pad, pad, pad), mode="reflect")
    windows = patch.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    return windows.flatten(-2).median(dim=-1).values


def resize(image, size):
    return F.interpolate(image, size=tuple(size), mode="bilinear", align_corners=False)


def render_patch(patch, kernel_size, screen_size):
    return resize(median_pool(patch, kernel_size), screen_size)


def apply_patch(image, patch, box):
    x, y, width, height = box
    h, w = image.shape[-2:]
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > w or y + height > h:
        raise ValueError(f"Patch box {box} lies outside the {w}x{h} canvas")
    scaled = resize(patch, (height, width))
    padding = (x, w - x - width, y, h - y - height)
    mask = F.pad(torch.ones_like(scaled[:, :1]), padding)
    return image * (1 - mask) + F.pad(scaled, padding)


def total_variation(patch):
    dx = patch[:, :, :-1, :-1] - patch[:, :, :-1, 1:]
    dy = patch[:, :, :-1, :-1] - patch[:, :, 1:, :-1]
    return torch.linalg.vector_norm(torch.stack((dx, dy)), dim=0).sum()


def quantize(image):
    return image.detach().clamp(0, 1).mul(255).round().div(255)
