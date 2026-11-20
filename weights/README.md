# Checkpoints

| Configured file | Backend / architecture | Input convention in supplied config |
|---|---|---|
| `arcface_r100.pth` | InsightFace / `iresnet100` | RGB, `(x - 0.5) / 0.5` |
| `cosface_r100.pth` | InsightFace / `iresnet100` | RGB, `(x - 0.5) / 0.5` |
| `magface_epoch_00025.pth` | Original MagFace / `iresnet100` | BGR, `[0,1]` |

## Sources

- [InsightFace ArcFace PyTorch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch): backbone code, training implementation and model resources.
- [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo): inspect each release's framework/architecture before downloading. ONNX and MXNet models are **not** interchangeable with PyTorch state dictionaries.
- [MagFace author repository](https://github.com/IrvingMeng/MagFace): pretrained model table, including original MS1MV2 IResNet100 weights. The provided MagFace preprocessing follows the original `inference/gen_feat.py`, which reads BGR with OpenCV and uses mean 0/std 1. Other reimplementations may use different conventions.

For CosFace, supply a PyTorch IResNet checkpoint actually trained with the CosFace margin loss, or train one using the InsightFace training code. This archive does not provide an unverified CosFace download link. A SphereFace, face.evoLVe `IR_100`, ONNX or MXNet checkpoint will not load into this IResNet backend. ArcFace/CosFace identify training losses, not a universal checkpoint format.

Do not reuse the same weights under different model names. The generation script rejects identical checkpoint bytes, but architecture checks and hashes cannot prove the training loss or dataset; verify provenance yourself. The notebook's `ir152.pth` and `ir152_ar.pth` cannot replace the paper's three-model ensemble.

## Supported state dictionaries

- A bare tensor state dictionary.
- A checkpoint containing `state_dict` or `model_state_dict`.
- Common prefixes: `module.`, `module.features.`, `features.module.`, `features.`, `backbone.`, `module.backbone.`.
- An explicit `state_prefix` can be set in the model config.

Every expected tensor, including BatchNorm buffers, must match by name and shape. The loader rejects incomplete matches instead of retaining randomly initialized parameters. Extra classification-head keys are ignored and recorded in `manifest.json`. Loading uses `torch.load(..., weights_only=True)`.

The provided MagFace and InsightFace backbones deliberately retain their different BatchNorm epsilon values. Checkpoint preprocessing is also separate. These differences must not be erased merely because the networks have similarly named layers.
