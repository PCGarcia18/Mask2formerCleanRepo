# Based on FaPN code by Shihua Huang et al.
import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch
from torch import nn
from torchvision.ops import deform_conv2d
from torch.cuda.amp import autocast

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

__all__ =["FaPN_Fusion_2026_PixelDecoder"]

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

class FeatureAlign_V2(nn.Module):
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        
        self.out_nc = out_nc
        self.deformable_groups = 8
        self.kernel_size = 3
        
        self.deform_conv_weight = nn.Parameter(torch.randn(out_nc, out_nc // self.deformable_groups, 3, 3))
        self.deform_conv_bias = nn.Parameter(torch.zeros(out_nc))
        
        offset_channels = 2 * self.kernel_size * self.kernel_size * self.deformable_groups
        mask_channels = self.kernel_size * self.kernel_size * self.deformable_groups
        
        self.offset_mask_conv = nn.Conv2d(out_nc * 2, offset_channels + mask_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
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
        offset_mask = self.offset_mask_conv(torch.cat([feat_arm, feat_up * 2], dim=1))
        
        offset_channels = 2 * self.kernel_size * self.kernel_size * self.deformable_groups
        offset = offset_mask[:, :offset_channels, :, :]
        mask = torch.sigmoid(offset_mask[:, offset_channels:, :, :])
        
        batch_size, _, height, width = feat_up.shape
        channels_per_group = self.out_nc // self.deformable_groups
        
        feat_up_groups = feat_up.view(batch_size, self.deformable_groups, channels_per_group, height, width)
        offset_per_group = 2 * self.kernel_size * self.kernel_size
        mask_per_group = self.kernel_size * self.kernel_size
        
        output_groups =[]
        for g in range(self.deformable_groups):
            feat_group = feat_up_groups[:, g, :, :, :].contiguous()
            offset_group = offset[:, g*offset_per_group:(g+1)*offset_per_group, :, :].contiguous()
            mask_group = mask[:, g*mask_per_group:(g+1)*mask_per_group, :, :].contiguous()
            
            weight_group = self.deform_conv_weight[g*channels_per_group:(g+1)*channels_per_group, :, :, :]
            bias_group = self.deform_conv_bias[g*channels_per_group:(g+1)*channels_per_group]
            
            output_group = deform_conv2d(feat_group, offset_group, weight_group, bias_group, stride=(1, 1), padding=(1, 1), dilation=(1, 1), mask=mask_group)
            output_groups.append(output_group)
        
        feat_align = torch.cat(output_groups, dim=1)
        feat_align = self.relu(feat_align)
        
        return feat_align + feat_arm


class SingleBranchFaPN(nn.Module):
    """Encapsula la lógica de un FaPN para poder instanciarlo dos veces."""
    def __init__(self, feature_channels, conv_dim, norm):
        super().__init__()
        self.scale_heads = nn.ModuleList()
        
        # Build alignment modules for each scale
        for idx in range(len(feature_channels) - 1):
            in_channels = feature_channels[idx]
            self.scale_heads.append(FeatureAlign_V2(in_channels, conv_dim, norm=get_norm(norm, conv_dim)))
        
        # Last feature level gets a simple lateral connection
        self.scale_heads.append(
            Conv2d(feature_channels[-1], conv_dim, kernel_size=1, bias=norm == "", norm=get_norm(norm, conv_dim))
        )
        
        # Output convolutions for each FPN level
        self.output_convs = nn.ModuleList()
        for idx in range(len(feature_channels) - 1):
            output_conv = Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=norm == "", norm=get_norm(norm, conv_dim))
            weight_init.c2_xavier_fill(output_conv)
            self.output_convs.append(output_conv)

    def forward(self, features_list):
        results =[]
        # Start from the smallest (highest semantic level)
        prev_features = self.scale_heads[-1](features_list[-1])
        results.append(prev_features)
        
        for idx in range(len(features_list) - 2, -1, -1):
            prev_features = self.scale_heads[idx](features_list[idx], prev_features)
            if idx < len(self.output_convs):
                prev_features = self.output_convs[idx](prev_features)
            results.insert(0, prev_features)
            
        return results  # Retorna lista de tensores[res2, res3, res4, res5]


# class FeatureFusionModule(nn.Module):
#     """Módulo para fusionar las salidas de la rama RGB y MSI."""
#     def __init__(self, conv_dim, norm, num_scales=4):
#         super().__init__()
        
#         # Fusion para las escalas bajas (Transformer Decoder) y alta (Dot product)
#         self.fusion_blocks = nn.ModuleList()
#         for i in range(num_scales):
#             if i == 0:
#                 # El de mayor resolución (res2) requiere contexto espacial extra (3x3)
#                 block = nn.Sequential(
#                     Conv2d(conv_dim * 2, conv_dim, kernel_size=3, padding=1, bias=norm == "", norm=get_norm(norm, conv_dim)),
#                     nn.ReLU(inplace=True),
#                     Conv2d(conv_dim, conv_dim, kernel_size=1, bias=norm == "", norm=get_norm(norm, conv_dim))
#                 )
#                 weight_init.c2_xavier_fill(block[0])
#                 weight_init.c2_xavier_fill(block[2])
#             else:
#                 # Los multiescala (res3, res4, res5) bajan dimensionalidad con 1x1
#                 block = Conv2d(conv_dim * 2, conv_dim, kernel_size=1, bias=norm == "", norm=get_norm(norm, conv_dim))
#                 weight_init.c2_xavier_fill(block)
                
#             self.fusion_blocks.append(block)

#     def forward(self, fpn_rgb_list, fpn_msi_list):
#         fused_list =[]
#         for i in range(len(fpn_rgb_list)):
#             concat_feat = torch.cat([fpn_rgb_list[i], fpn_msi_list[i]], dim=1)
#             fused_list.append(self.fusion_blocks[i](concat_feat))
#         return fused_list



class SpatialAttentionFusionBlock(nn.Module):
    def __init__(self, conv_dim, norm, use_3x3=False):
        super().__init__()
        
        kernel_attn = 3 if use_3x3 else 1
        pad_attn = 1 if use_3x3 else 0
        
        self.attention_generator = nn.Sequential(
            Conv2d(conv_dim * 2, conv_dim // 2, kernel_size=kernel_attn, padding=pad_attn),
            nn.ReLU(inplace=True),
            Conv2d(conv_dim // 2, 2, kernel_size=1), # Salida: 2 canales espaciales
            nn.Sigmoid() # Fuerza los pesos a estar entre 0 y 1
        )
        
        proj_kernel = 3 if use_3x3 else 1
        proj_pad = 1 if use_3x3 else 0
        self.proj = Conv2d(conv_dim * 2, conv_dim, kernel_size=proj_kernel, padding=proj_pad, bias=norm == "", norm=get_norm(norm, conv_dim))
        
        weight_init.c2_xavier_fill(self.attention_generator[0])
        weight_init.c2_xavier_fill(self.attention_generator[2])
        weight_init.c2_xavier_fill(self.proj)

    def forward(self, rgb, msi):
        concat_feat = torch.cat([rgb, msi], dim=1)
        
        attn_weights = self.attention_generator(concat_feat)
        
        weight_rgb = attn_weights[:, 0:1, :, :] # [Batch, 1, H, W]
        weight_msi = attn_weights[:, 1:2, :, :] #[Batch, 1, H, W]
        
        # Multiplicar cada rama por su mapa de atención espacial
        w = weight_msi[0, 0]
        total = w.numel()
        high  = (w > 0.7).sum().item() / total * 100
        mid   = ((w > 0.3) & (w <= 0.7)).sum().item() / total * 100
        low   = (w <= 0.3).sum().item() / total * 100
        print(f"  [{w.shape[0]}x{w.shape[1]}] "
            f"MSI high(>0.7): {high:.1f}%  "
            f"mid(0.3-0.7): {mid:.1f}%  "
            f"low(<0.3): {low:.1f}%  "
            f"(mean RGB={weight_rgb[0,0].mean().item():.3f} "
            f"MSI={w.mean().item():.3f})")


        rgb_attended = rgb * weight_rgb
        msi_attended = msi * weight_msi
        
        fused = torch.cat([rgb_attended, msi_attended], dim=1)
        out = self.proj(fused)
        
        return out


class FeatureFusionModule(nn.Module):
    """Módulo de Fusión Dinámica con Atención Espacial."""
    def __init__(self, conv_dim, norm, num_scales=4):
        super().__init__()
        
        self.fusion_blocks = nn.ModuleList()
        for i in range(num_scales):
            if i == 0:
                block = SpatialAttentionFusionBlock(conv_dim, norm, use_3x3=True)
            else:
                block = SpatialAttentionFusionBlock(conv_dim, norm, use_3x3=False)
                
            self.fusion_blocks.append(block)

    def forward(self, fpn_rgb_list, fpn_msi_list):
        fused_list =[]
        for i in range(len(fpn_rgb_list)):
            fused_feat = self.fusion_blocks[i](fpn_rgb_list[i], fpn_msi_list[i])
            fused_list.append(fused_feat)
        return fused_list




@SEM_SEG_HEADS_REGISTRY.register()
class FaPN_Fusion_2026_PixelDecoder(nn.Module):
    """
    Dual-Branch Feature Alignment Pyramid Network con Fusión Tardía.
    """
    def __init__(self, cfg, input_shape: dict):
        super().__init__()

        # "res2", "res3", "res4", "res5"
        self.base_in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM

        # Obtenemos los canales mirando solo las keys de RGB (las MSI son idénticas)
        feature_channels = [input_shape[f"{f}_rgb"].channels for f in self.base_in_features]
        
        # Instanciamos los dos FaPN paralelos
        self.fapn_rgb = SingleBranchFaPN(feature_channels, conv_dim, norm)
        self.fapn_msi = SingleBranchFaPN(feature_channels, conv_dim, norm)
        
        self.fusion_module = FeatureFusionModule(conv_dim, norm, num_scales=len(self.base_in_features))
        
        self.mask_features_head = nn.Sequential(
            Conv2d(conv_dim, mask_dim, kernel_size=3, stride=1, padding=1, bias=norm == "", norm=get_norm(norm, mask_dim)),
            nn.ReLU(inplace=True)
        )
        weight_init.c2_xavier_fill(self.mask_features_head[0])

    @autocast(enabled=False)
    def forward_features(self, features):
        """
        Args:
            features (dict[str->Tensor]): Contiene 'res2_rgb', 'res2_msi', etc.
        """
        x_rgb = [features[f"{f}_rgb"] for f in self.base_in_features]
        x_msi = [features[f"{f}_msi"] for f in self.base_in_features]
        
        fpn_rgb_out = self.fapn_rgb(x_rgb)
        fpn_msi_out = self.fapn_msi(x_msi)
        
        fused_features = self.fusion_module(fpn_rgb_out, fpn_msi_out)
        
        mask_features = self.mask_features_head(fused_features[0])
        
        # Los features restantes (stride 8, 16, 32) alimentan al Transformer Decoder
        if len(fused_features) >= 3:
            multi_scale_features = fused_features[-3:]
        else:
            multi_scale_features = fused_features
            
        return mask_features, None, multi_scale_features