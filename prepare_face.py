import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


TEMPLATE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                         [56.0252, 71.7366], [41.5493, 92.3655],
                         [70.7299, 92.2041]])


def similarity_transform(source, target):
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.shape != (5, 2) or target.shape != (5, 2):
        raise ValueError("Expected five (x, y) landmarks")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Landmarks must be finite")
    x, y = source.T
    design = np.zeros((10, 4))
    design[0::2] = np.column_stack((x, -y, np.ones(5), np.zeros(5)))
    design[1::2] = np.column_stack((y, x, np.zeros(5), np.ones(5)))
    params, _, rank, _ = np.linalg.lstsq(design, target.reshape(-1), rcond=None)
    if rank != 4:
        raise ValueError("Degenerate landmark configuration")
    a, b, tx, ty = params
    if a * a + b * b < 1e-12:
        raise ValueError("Degenerate alignment scale")
    return np.array([[a, -b, tx], [b, a, ty], [0., 0., 1.]])


def main():
    parser = argparse.ArgumentParser(description="Align a face using five supplied landmarks.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--landmarks", required=True, type=Path,
                        help="JSON array: left-image eye, right-image eye, nose, left mouth, right mouth")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output file already exists")
    if args.size < 1:
        parser.error("--size must be positive")
    source = np.asarray(json.loads(args.landmarks.read_text()), dtype=float)
    matrix = similarity_transform(source, TEMPLATE_112 * (args.size / 112))
    inverse = np.linalg.inv(matrix)
    with Image.open(args.image) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        aligned = image.transform((args.size, args.size), Image.Transform.AFFINE,
                                  tuple(inverse[:2].reshape(-1)), Image.Resampling.BILINEAR)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aligned.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
