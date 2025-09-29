import torch.nn as nn
import timm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class D2TIMMConvNeXt(Backbone):
    """
    ConvNeXt backbone for Detectron2 (Mask2Former). Uses timm for model loading.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        # Get model configs
        self.in_chans = len(cfg.MODEL.PIXEL_MEAN)  # Input channels
        #self.out_features = cfg.MODEL.CONVNEXT.OUT_FEATURES  # Features
        self.out_features = ["res2", "res3", "res4", "res5"]  # Features
        #self.pretrained = cfg.MODEL.CONVNEXT.PRETRAINED  # Pretrained weights imageNet1k
        self.pretrained = False

        
        # Create ConvNeXt model with timm
        self.convnext = timm.create_model(
            "convnext_tiny", 
            pretrained=self.pretrained,  # BOOL
            in_chans=self.in_chans,  
            features_only=True,  # Esto sirve para eliminar la classification head, asi se puede usar directamente como encoder
            #checkpoint_path = "/home/pablo.canosa/wip/Mask2Former/backbone_weights/ConvNext/onvnext_multi_imageNet1k_fine_tuned_modified_to_backbone.pth" # Opcional si quiero usar mis pesos
            checkpoint_path = "/home/pablo.canosa/wip/Mask2Former/backbone_weights/ConvNext/convnext_upernet_encoder_60ep.pth"
        )

        # Strides and channels
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            "res2": 96,  # Stage 1
            "res3": 192,  # Stage 2
            "res4": 384,  # Stage 3
            "res5": 768,  # Stage 4
        }

    def forward(self, x):
        """
        Forward pass of ConvNeXt backbone.
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            dict[str -> Tensor]: Feature maps at different scales.
        """
        features = self.convnext(x)  # Get feature maps from ConvNeXt
        # Convert feature maps to Detectron2 format
        outputs = {
            "res2": features[0],  # Output of first downsampling stage
            "res3": features[1],  # Output of second downsampling stage
            "res4": features[2],  # Output of third downsampling stage
            "res5": features[3],  # Output of fourth downsampling stage
        }
        #for output in outputs:
            #Print the shape of each output
            #print(outputs[output].shape)
        # Filter only requested output features
        return {k: v for k, v in outputs.items() if k in self.out_features}

    def output_shape(self):
        """
        Returns the shape of the output features.
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], 
                stride=self._out_feature_strides[name]
            )
            for name in self.out_features
        }

    @property
    def size_divisibility(self):
        return 32  # Matches the default Swin Transformer implementation
