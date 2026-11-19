# Third-party code

## InsightFace

- Source repository: https://github.com/deepinsight/insightface
- Revision: `c6c3af7bd2d0a0bf4b78c78437357f3521aa5d52`
- Original file: `recognition/arcface_torch/backbones/iresnet.py`
- Local file: `dipa/backbones/insightface.py`
- License: MIT, as declared by the source repository's README; a license notice is included in `dipa/backbones/LICENSE.insightface`.
- Local change: replaced deprecated `torch.cuda.amp.autocast(self.fp16)` with `torch.autocast(device_type="cuda", enabled=self.fp16)`. FP16 is disabled in the configured attack models.

The source README distinguishes the MIT code license from the separate terms for training data and pretrained weights. Obtain weights under their applicable upstream terms.

## MagFace

- Source repository: https://github.com/IrvingMeng/MagFace
- Revision: `99bae614ac2643b9694bf18e0c5645272ff6acfa`
- Original file: `models/iresnet.py`
- Local file: `dipa/backbones/magface.py` (unchanged).
- License: Apache-2.0; the upstream license is included in `dipa/backbones/LICENSE.magface`.

The original `inference/gen_feat.py` and `inference/network_inf.py` were consulted for preprocessing and checkpoint prefixes; they are not bundled. The configured BGR `[0,1]` input follows that original inference implementation.
