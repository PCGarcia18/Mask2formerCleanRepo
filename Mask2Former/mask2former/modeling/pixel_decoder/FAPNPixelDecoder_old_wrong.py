# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch
from torch import nn
import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.ops import deform_conv2d #Hace falta para evitar instalar DCNv2
import torch.nn as nn
import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map

from torch.cuda.amp import autocast

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

__all__ = ["FAN"]

# from dcn_v2 import DCN, DCNPooling, DCNv2, DCNv2Pooling, dcn_v2_conv, dcn_v2_pooling
#from dcn_v2 import DCN as dcn_v2 # NECESITO ver como arreglar esto

from detectron2.layers import (CNNBlockBase, Conv2d, DeformConv, ModulatedDeformConv, ShapeSpec, get_norm, )

class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
        weight_init.c2_xavier_fill(self.conv_atten)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


#class FeatureAlign_V2(nn.Module):  # FaPN full version # This is the old version with DCNv2 dependency
#    def __init__(self, in_nc=128, out_nc=128, norm=None):
#        super(FeatureAlign_V2, self).__init__()
#        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
#        self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
#        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
#                                extra_offset_mask=True)
#        self.relu = nn.ReLU(inplace=True)
#        weight_init.c2_xavier_fill(self.offset)
#
#    def forward(self, feat_l, feat_s, main_path=None):
#        HW = feat_l.size()[2:]
#        if feat_l.size()[2:] != feat_s.size()[2:]:
#            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
#        else:
#            feat_up = feat_s
#        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
#        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
#        feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
#        return feat_align + feat_arm

class FeatureAlign_V2(nn.Module):  # FaPN full version, this one without DCNv2 dependency
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        
        # Deformable convolution parameters
        self.out_nc = out_nc
        self.deformable_groups = 8
        self.kernel_size = 3
        
        # Create convolution weight for deformable conv
        # Shape: [out_channels, in_channels_per_group, kernel_h, kernel_w]
        self.deform_conv_weight = nn.Parameter(torch.randn(out_nc, out_nc // self.deformable_groups, 3, 3))
        self.deform_conv_bias = nn.Parameter(torch.zeros(out_nc))
        
        # For offset and mask generation (DCNv2 with extra_offset_mask=True)
        # Each deformable group needs its own offset and mask
        # offset: 2 * kernel_size^2 * deformable_groups
        # mask: kernel_size^2 * deformable_groups
        offset_channels = 2 * self.kernel_size * self.kernel_size * self.deformable_groups  # 2*3*3*8 = 144
        mask_channels = self.kernel_size * self.kernel_size * self.deformable_groups        # 3*3*8 = 72
        
        self.offset_mask_conv = nn.Conv2d(
            out_nc * 2, 
            offset_channels + mask_channels,  # 144 + 72 = 216 channels
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.deform_conv_weight)
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
            
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        
        # Generate offset and mask
        offset_mask = self.offset_mask_conv(torch.cat([feat_arm, feat_up * 2], dim=1))
        
        # Split offset and mask
        offset_channels = 2 * self.kernel_size * self.kernel_size * self.deformable_groups  # 144
        offset = offset_mask[:, :offset_channels, :, :]  # [batch, 144, H, W]
        mask = offset_mask[:, offset_channels:, :, :]     # [batch, 72, H, W]
        
        # Apply sigmoid to mask (as typically done in DCNv2)
        mask = torch.sigmoid(mask)
        
        # Apply group-wise deformable convolution
        batch_size, _, height, width = feat_up.shape
        channels_per_group = self.out_nc // self.deformable_groups
        
        # Split input into groups
        feat_up_groups = feat_up.view(batch_size, self.deformable_groups, channels_per_group, height, width)
        
        # Split offset and mask for each group
        offset_per_group = 2 * self.kernel_size * self.kernel_size  # 18
        mask_per_group = self.kernel_size * self.kernel_size        # 9
        
        output_groups = []
        
        for g in range(self.deformable_groups):
            # Extract group-specific data
            feat_group = feat_up_groups[:, g, :, :, :].contiguous()  # [batch, channels_per_group, H, W]
            
            # Extract group-specific offset and mask
            offset_start = g * offset_per_group
            offset_end = (g + 1) * offset_per_group
            offset_group = offset[:, offset_start:offset_end, :, :].contiguous()
            
            mask_start = g * mask_per_group
            mask_end = (g + 1) * mask_per_group
            mask_group = mask[:, mask_start:mask_end, :, :].contiguous()
            
            # Extract group-specific weights and bias
            weight_group = self.deform_conv_weight[g*channels_per_group:(g+1)*channels_per_group, :, :, :]
            bias_group = self.deform_conv_bias[g*channels_per_group:(g+1)*channels_per_group] if self.deform_conv_bias is not None else None
            
            # Apply deformable convolution for this group
            output_group = deform_conv2d(
                input=feat_group,
                offset=offset_group,
                weight=weight_group,
                bias=bias_group,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                mask=mask_group
            )
            
            output_groups.append(output_group)
        
        # Concatenate all group outputs
        feat_align = torch.cat(output_groups, dim=1)  # [batch, out_nc, H, W]
        feat_align = self.relu(feat_align)
        
        return feat_align + feat_arm

#@SEM_SEG_HEADS_REGISTRY.register() #class FAN(Backbone): # Antes era así
class FAN(nn.Module):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FAN, self).__init__()

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        align_modules = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature[:-1]):
            stage = int(math.log2(strides[idx]))
            lateral_norm = get_norm(norm, out_channels)
            align_module = FeatureAlign_V2(in_channels, out_channels, norm=lateral_norm)  # proposed fapn
            self.add_module("fan_align{}".format(stage), align_module)
            align_modules.append(align_module)
            output_conv = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                 norm=get_norm(norm, out_channels), )
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)
            output_convs.append(output_conv)
        stage = int(math.log2(strides[len(in_channels_per_feature) - 1]))
        lateral_conv = Conv2d(in_channels_per_feature[-1], out_channels, kernel_size=1, bias=use_bias,
                              norm=get_norm(norm, out_channels))
        align_modules.append(lateral_conv)
        self.add_module("fan_align{}".format(stage), lateral_conv)
        # Place convs into top-down order (from low to high resolution) to make the top-down computation in forward clearer.
        self.align_modules = align_modules[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.align_modules[0](x[0])
        results.append(prev_features)
        for features, align_module, output_conv in zip(x[1:], self.align_modules[1:], self.output_convs[0:]):
            prev_features = align_module(features, prev_features)
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]) for
                name in self._out_features}

#    Tengo que hacer un forward_features para que devuelva lo siguiente:
#    Note that, your pixel decoder must have a self.forward_features(features) methods that returns three values:
#
#    mask_features, which is the per-pixel embeddings with resolution 1/4 of the original image. This is used to produce binary masks.
#    None, you can simply return None for the second value.
#    multi_scale_features, which is the multi-scale inputs to the Transformer decoder. This must be a list with length 3. We use resolution 1/32, 1/16, and 1/8 but you can use arbitrary resolutions here.

    @autocast(enabled=False)
    def forward_features(self, features):
        # Hay que completar este método para que se integre con Mask2Former

        return self.mask_features(out[-1]), None, multi_scale_features
    


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(stride, strides[i - 1])

