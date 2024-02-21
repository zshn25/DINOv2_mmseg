#
# COPYRIGHT (c) 2024 - Denso ADAS Engineering Services GmbH, Apache License 2.0
# Author: Zeeshan Khan Suri (z.suri@eu.denso.com)
#
# Wrapper for DINOv2 Vision Transformer backbone which is compatible with mmsegmentation >= 1.0

import sys
# Change this to your local path to DINOv2
sys.path.insert(0, "../../prototyping_dinov2")
from dinov2.models.vision_transformer import DinoVisionTransformer, Block, MemEffAttention

from typing import Optional
from functools import partial
import warnings

import torch
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class DinoVisionBackbone(DinoVisionTransformer, BaseModule):
    """mmsegmentation compatible Vision Transformer backbone.

    Inputs:
        size (str): size of ViT backbone. 'small', 'base', 'large', 'giant'
        freeze_vit (bool): Freezes the entire backbone.
            Default: False
        pretrained (str, optional): model pretrained path. (deprecated, use init_cfg instead)
            Default: None.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
        args, kwargs: Additional args that are passed to DinoVisionTransformer
    """
    # out_indices come from the DINOv2 configs, \
    # for e.g. https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_voc2012_ms_config.py
    out_indices = dict(small=[8, 9, 10, 11],
                       base=[8, 9, 10, 11],
                       large=[20, 21, 22, 23],
                       giant=[36, 37, 38, 39])

    def __init__(self,
                 size: str = "base",
                 freeze_vit: bool = False,
                 pretrained: Optional[str] = None,  # deprecated
                 init_cfg: Optional[dict] = None,
                 *args, 
                 **kwargs):

        # Update DinoVisionTransformer arguments based on model size
        if "small" in size:
            vit_kwargs = dict(embed_dim=384, depth=12, num_heads=6)
        elif "base" in size:
            vit_kwargs = dict(embed_dim=768, depth=12, num_heads=12)
        elif "large" in size:
            vit_kwargs = dict(embed_dim=1024, depth=24, num_heads=16)
        elif "giant" in size:
            vit_kwargs = dict(embed_dim=1536, depth=40, num_heads=24)
        else:
            raise NotImplementedError("Choose size from 'small', 'base', 'large', 'giant'")

        kwargs.update(**vit_kwargs)

        # Backward compatibility of the pretrained argument
        assert not (init_cfg and pretrained), \
            f'init_cfg: {init_cfg} and pretrained: {pretrained}, cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        super(DinoVisionBackbone, self).__init__(
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
            mlp_ratio=4,
            block_fn=partial(Block,
                             attn_class=MemEffAttention),
            *args, **kwargs)

        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

        self.out_index = self.out_indices[size]

        BaseModule.init_weights(self)  # explicitly call BaseModule's init_weights as both parent classes have the same named fn

        if freeze_vit:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, *args, **kwargs):
        return self.get_intermediate_layers(x=x,
                                            n=self.out_index,
                                            reshape=True,
                                            *args,
                                            **kwargs)
