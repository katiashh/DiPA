# Method and implementation choices

## Objective

For a clean face `x`, a square RGB patch parameter `p` in `[0, 1]`, and the frozen models `F_i`, the implementation minimizes

```math
L(p) = \sum_{i=1}^{3} \cos(F_i(A(x, M(p))), F_i(x)) + \lambda_{TV} TV(p).
```

Here `M` is channel-wise median pooling, and `A` renders the pooled patch at the screen resolution, resizes it into a fixed rectangle, and replaces that rectangle in the image. Each model then resizes the composed image to 112 × 112 and applies its own color order and normalization. Clean reference embeddings are cached once. There is no separate target identity.

`lambda_TV=0` gives DiPA. DiPA+TV uses a positive coefficient. The TV term is

```math
TV(p)=\sum_{c,i,j}\sqrt{(p_{c,i,j}-p_{c,i+1,j})^2+(p_{c,i,j}-p_{c,i,j+1})^2}.
```

Both differences are evaluated on the common interior of the patch; the channel-wise terms are summed. This is the explicit RGB/boundary convention used for Eq. (2). There is no division by pixel count. `torch.linalg.vector_norm` provides a finite zero subgradient at a flat patch, without adding an epsilon to the stated value.

Gradients pass through median pooling to the selected median values, through bilinear resize and through every frozen encoder to the patch. No straight-through estimator is used. Each model's contribution is backpropagated separately to reduce graph memory, and the gradients are accumulated before one Adam step. Pixels are clamped to `[0, 1]` after each step. Model weights and clean pixels are never updated.

## What is specified and what is reconstructed

| Component | Paper | This repository |
|---|---|---|
| Attack goal | Dodging, same clean identity | Minimize cosine with `F_i(x)` |
| Surrogates | ArcFace, CosFace, MagFace | Exactly these three names required during generation; different checkpoint hashes required |
| Aggregation | Sum in Eq. (1) | Unweighted sum, not mean |
| Patch processing | `MedianPool(p)` | Channel-wise 3 × 3, stride 1, reflection padding, followed by resize |
| Parameter shape | `D × D × 3` | 512 × 512 × 3 by default |
| Input to FR models | 3 × 112 × 112 | 112 × 112 per model |
| Placement | Near the mouth | Fixed lower-face box from the notebook's loop |
| Variants | No TV / TV | `configs/dipa.json` / `configs/dipa_tv.json` |
| Backbone depths, weights | Not specified | IResNet100 examples; weights supplied separately |
| Optimizer and learning rate | Not specified | Adam, 0.03 |
| Iteration count | Not specified | 200; inherited count from the notebook, not a paper claim |
| TV coefficient | Not specified | 1e-5 for **summed** TV; a configurable starting value |
| Median kernel, stride, padding | Not specified | Explicit defaults above, not a recovered original setting |
| Initialization | Not specified | Uniform RGB noise, independent seeds |
| Digital placement/canvas sizes | Not specified | Canvas 256 × 256; box `[45,160,160,90]`; screen 512 × 288 |
| Preprocessing | Not specified | Explicit per-checkpoint configuration |
| EOT/physical transformations | Not described | None |

The general prose introduces an enrolled reference `x_ref`, while Eq. (1) uses the same input `x`. This implementation follows Eq. (1). A different enrollment photograph is not silently substituted.

Median pooling alone is not claimed to map every patch directly to a 112 × 112 model input. The repository uses size-preserving local median pooling, then explicit resampling through the compositor and encoder. That choice is documented because the paper does not define the pooling operator in sufficient detail to recover it exactly.

## Changes from the notebook

| Notebook | Script implementation |
|---|---|
| Sibling-Attack with IR-152 FR and an attribute model | ArcFace + CosFace + MagFace ensemble |
| Alternating FR/AR inner loops and hooks | A single patch objective from Eq. (1) |
| Minimizes `1 - cosine` with `victim_img` | Minimizes cosine with the same clean face |
| Average pooling of the whole image | Median pooling of the patch only |
| Perturbation projected relative to the original face | Independent RGB patch bounded to `[0,1]` |
| No TV implementation | Optional isotropic TV from Eq. (2) |
| `attack_img = ori_attack_img` followed by in-place insertion | Functional compositing; clean reference stays unchanged |
| Initial crop `(48,165)` differs from loop placement `(45,160)` | One explicit box used for the whole run |
| Colab paths, execution-order state, undeclared wildcard imports | Standalone scripts, explicit dependencies and paths |
| Adversarial face output only | Display patch, composed image, loss history and run metadata |

The notebook's de-normalization multiplies by `mean` and adds `std`; it only works incidentally when they are equal. The new implementation keeps images in `[0,1]` and normalizes inside each encoder, so export needs no inverse model normalization.

## Camera metrics

Eq. (3) treats the camera output as an identity label. This code uses `identity_change_rate` for the fraction of eligible trials whose patched label differs from the clean label. Empty labels are considered different. A detector failure can therefore increase that rate; it is not evidence of recognition-only dodging.

The script also reports:

- `detection_rate`: detected patched faces / eligible trials.
- `detected_dodging_rate`: detected faces with changed identity / eligible trials.
- `dodging_rate_given_detection`: detected faces with changed identity / detected faces.
- `mean_face_confidence`: mean camera face-presence confidence across eligible trials.

Confidence values retain the camera's scale. Missing detection or identity values must not be inferred from cosine similarity. Rows without successful clean verification are excluded and counted. No commercial-camera threshold or recognition label is synthesized from local embeddings.

## Reproducibility limits

The original trained weights, exact configuration, volunteer images and camera trial records were not included in the supplied files. This repository reconstructs the described objective and exposes missing implementation choices. It does not claim numerical reproduction of Table I, physical ASR, detection confidence, or the original web service. Matching the published setup requires the authors' checkpoint provenance, preprocessing, placement, display/camera settings and trial data.
