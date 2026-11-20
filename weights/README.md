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
