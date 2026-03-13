# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

# Para pruebas de visualización
import os
import cv2
from detectron2.utils.visualizer import Visualizer

__all__ = ["MaskFormerSemanticDatasetMapperAugmented"]


class MaskFormerSemanticDatasetMapperAugmented:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        
        augs.append(T.RandomRotation(angle=[-45, 45], expand=False)) # Probar con esto angle=[-180, 180]
        
        

        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))

        augs.append(T.RandomFlip())
        
        # Transformadas extras no probadas
        augs.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True)) #Default es en horizontal, asi que añadimos esta

        


        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapperAugmented should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)

        # Read custom images
        if(dataset_dict["NIR"]):
            image = utils.read_rawb_NirRGB(dataset_dict["file_name"])
        else:
            image = utils.read_rawb_RGB(dataset_dict["file_name"], dataset_dict["DATASET_NAME"])

        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # -------------------------------------------------------------------------
        # Guardar imágenes originales antes de la transformación para comparación visual
        # -------------------------------------------------------------------------
        image_original_vis = image.copy()
        mask_original_vis = sem_seg_gt.copy() if sem_seg_gt is not None else None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        DEBUG_SAVE_IMAGES = False 
        OUTPUT_DEBUG_DIR = "./debug_augmentations"

        def convert_for_vis(img_array):
            """
            Convierte cualquier imagen (uint16, float, etc.) a uint8 0-255 
            para poder visualizarla correctamente con OpenCV.
            """
            vis = img_array[:, :, :].copy()
            
            if vis.max() > 255 or vis.dtype == np.uint16 or vis.dtype == np.float32:
                vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
            
            vis = vis.astype("uint8")
            
            # Detectron carga en RGB (o BGR dependiendo del loader), 
            # pero OpenCV guarda en BGR. 
            vis = vis[:, :, ::-1] 
            
            return vis

        if DEBUG_SAVE_IMAGES:
            os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)
            filename_base = os.path.basename(dataset_dict["file_name"]).split('.')[0]
            
            vis_img_orig = convert_for_vis(image_original_vis)
            cv2.imwrite(f"{OUTPUT_DEBUG_DIR}/{filename_base}_orig_img.jpg", vis_img_orig)
            
            if mask_original_vis is not None:
                cv2.imwrite(f"{OUTPUT_DEBUG_DIR}/{filename_base}_orig_mask.png", mask_original_vis.astype("uint8"))

            vis_img_aug = convert_for_vis(image)

            if sem_seg_gt is not None:
                mask_aug_save = sem_seg_gt.astype("uint8")
                
                cv2.imwrite(f"{OUTPUT_DEBUG_DIR}/{filename_base}_aug_mask_vis.png", mask_aug_save)
                

            # Intentar usar Visualizer de Detectron2
            try:
                vis_img_for_detectron = vis_img_aug[:, :, ::-1] # Invertimos canales para Visualizer (BGR->RGB)
                
                dataset_name = dataset_dict.get("DATASET_NAME")
                meta = MetadataCatalog.get(dataset_name) if dataset_name else None
                
                v_aug = Visualizer(vis_img_for_detectron, metadata=meta, scale=1.0)
                
                if sem_seg_gt is not None:
                    v_aug = v_aug.draw_sem_seg(sem_seg_gt.astype("int"))
                    cv2.imwrite(f"{OUTPUT_DEBUG_DIR}/{filename_base}_aug_overlay.jpg", v_aug.get_image()[:, :, ::-1])
                
                cv2.imwrite(f"{OUTPUT_DEBUG_DIR}/{filename_base}_aug_img.jpg", vis_img_aug)

            except Exception as e:
                print(f"Warning visualizando {filename_base}: {e}")
                cv2.imwrite(f"{OUTPUT_DEBUG_DIR}/{filename_base}_aug_img_raw.jpg", vis_img_aug)



        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            classes = classes.flatten() 
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict