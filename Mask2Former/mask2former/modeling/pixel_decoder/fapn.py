# Based on FaPN code by Shihua Huang et al.
# Paper: FaPN: Feature-aligned Pyramid Network for Dense Image Prediction
# ArXiv: https://arxiv.org/pdf/2108.07058
# Github repo: https://github.com/EMI-Group/FaPN

import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch
from torch import nn
from torchvision.ops import deform_conv2d
from torch.cuda.amp import autocast

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY  # or use SEM_SEG_HEADS_REGISTRY if needed

__all__ = ["FaPN"]


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


class FeatureAlign_V2(nn.Module):  # FaPN full version without DCNv2 dependency
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        
        # Deformable convolution parameters
        self.out_nc = out_nc
        self.deformable_groups = 8
        self.kernel_size = 3
        
        # Create convolution weight for deformable conv
        self.deform_conv_weight = nn.Parameter(torch.randn(out_nc, out_nc // self.deformable_groups, 3, 3))
        self.deform_conv_bias = nn.Parameter(torch.zeros(out_nc))
        
        # For offset and mask generation
        offset_channels = 2 * self.kernel_size * self.kernel_size * self.deformable_groups  # 144
        mask_channels = self.kernel_size * self.kernel_size * self.deformable_groups        # 72
        
        self.offset_mask_conv = nn.Conv2d(
            out_nc * 2, 
            offset_channels + mask_channels,
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
            
        feat_arm = self.lateral_conv(feat_l)
        
        # Generate offset and mask
        offset_mask = self.offset_mask_conv(torch.cat([feat_arm, feat_up * 2], dim=1))
        
        # Split offset and mask
        offset_channels = 2 * self.kernel_size * self.kernel_size * self.deformable_groups
        offset = offset_mask[:, :offset_channels, :, :]
        mask = offset_mask[:, offset_channels:, :, :]
        
        # Apply sigmoid to mask
        mask = torch.sigmoid(mask)
        
        # Apply group-wise deformable convolution
        batch_size, _, height, width = feat_up.shape
        channels_per_group = self.out_nc // self.deformable_groups
        
        # Split input into groups
        feat_up_groups = feat_up.view(batch_size, self.deformable_groups, channels_per_group, height, width)
        
        # Split offset and mask for each group
        offset_per_group = 2 * self.kernel_size * self.kernel_size
        mask_per_group = self.kernel_size * self.kernel_size
        
        output_groups = []
        
        for g in range(self.deformable_groups):
            # Extract group-specific data
            feat_group = feat_up_groups[:, g, :, :, :].contiguous()
            
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
        feat_align = torch.cat(output_groups, dim=1)
        feat_align = self.relu(feat_align)
        
        return feat_align + feat_arm

@SEM_SEG_HEADS_REGISTRY.register()
class FaPNPixelDecoder(nn.Module):
    """
    Feature Alignment Pyramid Network as a pixel decoder for Mask2Former.
    """

    #def __init__(self, input_shape, conv_dim=256, mask_dim=256, norm="GN"):
    def __init__(self, cfg, input_shape:dict):
        """
        Args:
            input_shape (dict[str->ShapeSpec]): A dictionary of feature names to their shapes
                from the backbone. Should contain feature maps at different resolutions.
            conv_dim (int): Number of channels in the FPN outputs.
            mask_dim (int): Number of channels for the mask features output.
            norm (str): Normalization type to use.
        """
        super().__init__()

        # features to use (from config)
        in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        self.in_features = in_features
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM

        # filter only requested features
        feature_shapes = {k: v for k, v in input_shape.items() if k in in_features}
        print("Feature shapes used in FaPN:", feature_shapes)

        # sort by stride (feature_shapes.items() gives list of (name, ShapeSpec))
        feature_shapes = sorted(feature_shapes.items(), key=lambda x: x[1].stride)

        # extract names, strides, channels
        self.in_features = [k for k, v in feature_shapes]
        feature_strides = [v.stride for k, v in feature_shapes]
        feature_channels = [v.channels for k, v in feature_shapes]

        
        # Build FaPN
        self.scale_heads = nn.ModuleList()
        
        # Build alignment modules for each scale
        for idx in range(len(self.in_features) - 1):
            in_channels = feature_channels[idx]
            align_module = FeatureAlign_V2(in_channels, conv_dim, norm=get_norm(norm, conv_dim))
            self.scale_heads.append(align_module)
        
        # Last feature level gets a simple lateral connection
        self.scale_heads.append(
            Conv2d(
                feature_channels[-1],
                conv_dim,
                kernel_size=1,
                bias=norm == "",
                norm=get_norm(norm, conv_dim)
            )
        )
        
        # Output convolutions for each FPN level
        self.output_convs = nn.ModuleList()
        for idx in range(len(self.in_features) - 1):
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=norm == "",
                norm=get_norm(norm, conv_dim)
            )
            weight_init.c2_xavier_fill(output_conv)
            self.output_convs.append(output_conv)
        
        # Mask feature head - projects features to mask_dim for mask prediction
        self.mask_features_head = nn.Sequential(
            Conv2d(
                conv_dim,
                mask_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=norm == "",
                norm=get_norm(norm, mask_dim)
            ),
            nn.ReLU(inplace=True)
        )
        
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim

    def forward(self, features):
        """
        Args:
            features (dict[str->Tensor]): Feature maps from the backbone.
        
        Returns:
            dict[str->Tensor]: FPN feature maps.
        """
        x = [features[f] for f in self.in_features]
        
        # Build FPN top-down
        results = []
        
        # Start from the smallest
        prev_features = self.scale_heads[-1](x[-1])
        results.append(prev_features)
        
        for idx in range(len(x) - 2, -1, -1):
            align_idx = idx
            features = x[idx]
            prev_features = self.scale_heads[align_idx](features, prev_features)
            
            if idx < len(self.output_convs):
                prev_features = self.output_convs[idx](prev_features)
            
            results.insert(0, prev_features)
        
        out = {f: res for f, res in zip(self.in_features, results)}
        
        return out

    @autocast(enabled=False)
    def forward_features(self, features):
        """
        Forward method for Mask2Former compatibility.
        
        Args:
            features (dict[str->Tensor]): Feature maps from the backbone.
        
        Returns:
            tuple: (mask_features, None, multi_scale_features)
                - mask_features: Per-pixel embeddings at 1/4 resolution
                - None: Placeholder for transformer encoder features
                - multi_scale_features: List of 3 feature maps for transformer decoder
        """
        fpn_features = self.forward(features)

        feature_list = [fpn_features[k] for k in self.in_features]
        
        mask_features = self.mask_features_head(feature_list[0])
        

        if len(feature_list) >= 3:
            multi_scale_features = feature_list[-3:]
        else:
            multi_scale_features = feature_list
            
        return mask_features, None, multi_scale_features


#def _assert_strides_are_log2_contiguous(strides):
#    """
#    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
#    """
#    for i, stride in enumerate(strides[1:], 1):
#        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(stride, strides[i - 1])