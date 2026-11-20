# DiPA

Script-based implementation of **From Pixels to Reality: Physical-Digital Patch Attacks on Real-World Camera** (Leonenkova et al., 2026).

[Paper](https://ieeexplore.ieee.org/abstract/document/11585403) · [Method details](docs/method.md) · [Weights](weights/README.md) 

DiPA learns a personalized patch by minimizing the sum of cosine similarities between a patched face and the same clean face across ArcFace, CosFace and MagFace. Median pooling is applied to the patch before compositing. `DiPA+TV` adds isotropic total variation to the objective.

## Install

Python 3.10 or newer. Run commands from this repository's root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows activate with `.venv\Scripts\activate`. Install a CUDA-enabled PyTorch build for GPU execution using the [PyTorch installation guide](https://pytorch.org/get-started/locally/). CPU execution is supported but a three-model R100 ensemble is expensive.

## Model weights

Provide three **different, correctly trained** checkpoints:

```text
weights/arcface_r100.pth
weights/cosface_r100.pth
weights/magface_epoch_00025.pth
```

The first two must match the InsightFace PyTorch IResNet architecture. The last must match the original MagFace architecture. See [weights/README.md](weights/README.md) for sources, supported formats and preprocessing. Renaming an ArcFace checkpoint to CosFace does not create a CosFace model. Weights are not included or downloaded automatically.

## Prepare a face

Use a frontal, aligned RGB face crop. `generate.py` resizes the crop to the configured canvas; it does **not** detect or align faces automatically. Check that the configured patch box covers the intended lower-face region on your input.

If you have five landmarks, alignment is available as a separate script:

```bash
python prepare_face.py --image portrait.jpg --landmarks landmarks.json --output data/face.png
```

`landmarks.json` is a list of five `[x, y]` pairs in this order: image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner. Coordinates refer to the EXIF-oriented original image. The default output is 256 × 256. Landmark detection itself is not included.

## Generate patches

```bash
python generate.py --image data/face.png --config configs/dipa.json --output outputs/dipa --device cuda --num-patches 5
python generate.py --image data/face.png --config configs/dipa_tv.json --output outputs/dipa_tv --device cuda --num-patches 5
```

Seeds are `seed`, `seed + 1`, etc. Both variants use the same seeds by default. Use `--steps` or `--seed` to override the configuration. Output directories must be new or empty.

Each run saves:

- `manifest.json`: resolved run configuration, model settings, checkpoint hashes, input hash and software versions.
- `clean.png`: the clean canvas used for saved-image evaluation.
- `patch_000/patch.png`: the median-pooled, rendered **display patch** (512 × 288 by default).
- `patch_000/patched.png`: the clean face with the display patch inserted.
- `patch_000/latent_patch.png` and `patch.pt`: the optimized square parameter before median pooling; `patch.pt` retains float precision.
- `patch_000/history.csv`: each evaluated step, total objective, TV and per-model cosine similarities.
- `patch_000/metrics.json`: best step, actual seed, float metrics and metrics recomputed on exported PNGs.

Display `patch.png` on the phone, preserving its aspect ratio. Do not display `latent_patch.png` or apply median pooling a second time. The displayed patch corresponds to the lowest **total objective** encountered, including the TV term when enabled. Lower cosine similarity means stronger digital dodging; it does not determine camera ASR by itself.

The default patch box is `[x, y, width, height] = [45, 160, 160, 90]` on a 256 × 256 canvas. All sizes other than the box use `[height, width]`. Change the box to fit your image and keep `screen_size` at the same aspect ratio. A physical capture introduces geometry and camera processing that this digital composition does not simulate.

## Evaluate saved images

```bash
python evaluate.py --clean outputs/dipa/clean.png --patched outputs/dipa/patch_000/patched.png --config configs/dipa.json --output outputs/digital_scores.json --device cuda
```

`evaluate.py` accepts one or more supported local models via the `models` list. Using the same ensemble measures surrogate performance. Transfer evaluation requires an independently trained held-out model. Face++ requests, MobileFaceNet and a proprietary camera connection are not included; the provided files contain no corresponding implementation or credentials. This script scores prepared crops, without detecting and realigning the patched image.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover the objective direction, median gradients, TV, patch placement, checkpoint mapping, model input gradients, reproducible initialization, PNG export/evaluation and camera metric accounting. They use synthetic images and untrained/toy networks; they do not establish the attack success rates reported in the paper.

## Citation and third-party code

```bibtex
@INPROCEEDINGS{11585403,
  author={Leonenkova, Victoria and Shumitskaya, Ekaterina and Vatolin, Dmitriy and Antsiferova, Anastasia},
  booktitle={2026 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops)}, 
  title={From Pixels to Reality: Physical-Digital Patch Attacks on Real-World Camera}, 
  year={2026},
  volume={},
  number={},
  pages={412-414},
  keywords={Faces;Modeling;Cameras;Conferences;Face recognition;Printing;Smart phones;Displays;Pervasive computing;real-world camera;face recognition;adversarial patch;physical attack;digital attack;dodging attack},
  doi={10.1109/PerComWorkshops68308.2026.11585403}}
```

