# DINOv2_mmseg: Segmentation downstream of DINOv2 using mmsegmentation

Training and testing of DINOv2 for segmentation downstream


<p float="left">
  <img src="https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/images/Foggy_Cityscapes/dusseldorf_000081_000019_leftImg8bit_foggy_beta_0.02.png" width="49%" />
  <img src="assets/image.png?raw=true" width="49%" />
</p>

*Image source: [Semantic Foggy Scene Understanding with Synthetic Data](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)*


## ⚙️ Prerequisites
   
- Clone [DINOv2](https://github.com/facebookresearch/dinov2) repo and edit `train.py`, `test.py`, and `models/backbones/vit_dinov2.py` to change DINOv2's path. Specifically, replace the line `sys.path.append("../prototyping_dinov2")` with your path to DINOv2. This is required in particular if you get the error `KeyError: "EncoderDecoder: 'DinoVisionTransformer is not in the models registry'"`. 

  Alternatively, you can `python -m pip install -e <path-to-dinov2>` and remove the `sys.path.append("../prototyping_dinov2")` lines from all scripts.

- Install dependencies from DINOv2: https://github.com/facebookresearch/dinov2/blob/main/conda-extras.yaml

    ```bash
    conda env create -f conda.yaml
    conda activate dinov2
    ```

- Install mmcv>=2.0 and mmsegmentation >= 1.0. Check https://mmsegmentation.readthedocs.io/en/latest/get_started.html


## Training

1. An example config is provided for the Cityscapes dataset in `configs/dinov2_vitb14_cityscapes_ms_config.py`.
  Edit the config accoording to your preferences. Refer https://mmsegmentation.readthedocs.io/en/latest/user_guides/1_config.html for more details on the config.

  - To load the pretrained backbone and freeze it, change the config's `model.pretrained` to the path to pretrained backbone weights and `model.backbone.freeze_vit = True` to freeze it.

  - To load a complete (backbone + head) pretrained model, change the config's `load_from` to the path to the pretrained model file. Alternatively, `resume=True` with automatically find the last_checkpoint from your log_dir and resume training.

2. To run training, do `CUDA_VISIBLE_DEVICES=1,2,3 python train.py <path-to-config> `. Refer https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html



## Testing

3. To run test, do `CUDA_VISIBLE_DEVICES=1,2,3 python test.py <path-to-config> <path-to-checkpoint>`. Refer https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html


## Inference

Checkout segmentation.ipynb for an inference example.


# 👩‍⚖️ Licence


train.py modified from https://github.com/open-mmlab/mmsegmentation/blob/main/tools/train.py
test.py modified from https://github.com/open-mmlab/mmsegmentation/blob/main/tools/test.py


`train.py` and `test.py` are modified from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and therefore respect [mmsegmentation's licence](https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE).

The rest is licenced under Apache License 2.0, Copyright © Zeeshan Khan Suri, Denso ADAS Engineering Services GmbH, 2024.

