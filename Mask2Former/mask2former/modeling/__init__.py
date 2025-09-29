# Copyright (c) Facebook, Inc. and its affiliates.
from .backbone.swin import D2SwinTransformer

# Modificaciones
from .backbone.vitdet import D2ViTTransformer
from .backbone.swin_fusion import D2SwinTransformerFusionCNN
from .backbone.swin_fusion_v2 import D2SwinTransformerFusionCNNWithAttention
from .backbone.swin_fusion_doble import D2DualSwinTransformerFusion
from .backbone.swin_fusion_weighted_sum import D2SwinTransformerFusionCNNWeightedSum
from .backbone.swin_fusion_SE_Block import D2SwinTransformerSEBlockFusionCNN
from .backbone.swin_fusion_convnext import D2SwinTransformerConvNext
from .backbone.swin_fusion_conv_extractor_to_swin import D2SwinTransformerCNNChannelReduction
from .backbone.convnext import D2ConvNeXt
from .backbone.convnext_timm import D2TIMMConvNeXt

from .pixel_decoder.fpn import BasePixelDecoder
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .pixel_decoder.FAPNPixelDecoder_old_wrong import FAN
from .pixel_decoder.fapn import FaPNPixelDecoder

from .meta_arch.mask_former_head import MaskFormerHead
from .meta_arch.per_pixel_baseline import PerPixelBaselineHead, PerPixelBaselinePlusHead
