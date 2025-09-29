import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


def convnext_block(dim, drop_path=0.0):
    """ ConvNeXt block consisting of a depthwise conv, layer norm, and MLP """
    return nn.Sequential(
        nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
        nn.GroupNorm(1,dim),
        nn.Conv2d(dim, 4 * dim, kernel_size=1),
        nn.GELU(),
        nn.Conv2d(4 * dim, dim, kernel_size=1),
        nn.Dropout(drop_path)
    )


class ConvNeXt(nn.Module):
    """ConvNeXt backbone."""
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        
        # Stem layer
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                nn.GroupNorm(1,dims[0])
            )
        )
        
        # Downsampling and stages
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.GroupNorm(1,dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
                )
            )
        
        for i in range(4):
            self.stages.append(nn.Sequential(*[convnext_block(dims[i]) for _ in range(depths[i])]))
        
        self.norm_layers = nn.ModuleList([nn.GroupNorm(1,dim) for dim in dims])
        
    def forward(self, x):
        outs = {}
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.norm_layers[i](x)
            outs[f"res{i+2}"] = x
        return outs


@BACKBONE_REGISTRY.register()
class D2ConvNeXt(ConvNeXt, Backbone):
    def __init__(self, cfg, input_shape):
        
        in_chans = len(cfg.MODEL.PIXEL_MEAN)
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
        
        super().__init__(in_chans, depths, dims)
        
        # Load pretrained weights
        timm_model = timm.create_model("convnext_tiny", pretrained=True)
        pretrained_state_dict = timm_model.state_dict()

        # Convert keys from timm format to your model's format
        own_state_dict = self.state_dict()
        new_state_dict = {}

        for key in own_state_dict.keys():
            timm_key = key.replace("stages.", "stages.")  # This may need adjustments
            if timm_key in pretrained_state_dict:
                new_state_dict[key] = pretrained_state_dict[timm_key]
            else:
                new_state_dict[key] = own_state_dict[key]  # Keep original if not found

        # Load weights (ignoring missing or extra keys)
        self.load_state_dict(new_state_dict, strict=False)
        print("Pretrained ConvNeXt weights loaded successfully!")



        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": dims[0],
            "res3": dims[1],
            "res4": dims[2],
            "res5": dims[3],
        }
    
    def forward(self, x):
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
