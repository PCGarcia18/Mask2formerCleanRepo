# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified Mask2Former Training Script.
python train_net_gf_16bit_TIF_small.py --num-gpus 1 --config-file /home/pablo.canosa/wip/Mask2Former/configs/gaofen/sem-seg/swin/maskformer2_swin_tiny_bs16_50ep.yaml
This script is a simplified version of the training script in detectron2/tools. Is now adapted to Multispectral data.
"""
USE_NIR_BAND = False # Set to True if you want to use the NIR band in the multispectral images, else it will train on RGB images
DATASET_NAME = 'rios_red_init' # "rios_red_init"
APPLY_SQRT_SMOOTHING_TO_AUGMENTED_DATASET = True
# The images have to be created from the Five Billion Pixels with the jupyter notebook provided in the repository

#Train images path, use your own path
TRAIN_IMAGES_PATH = '/home/pablo.canosa/ssd/datasets_pablo/rios_FBP_BALANCED_REFERENCED_2026_validation/img_train/' 
TRAIN_PNG_MASKS_PATH = '/home/pablo.canosa/ssd/datasets_pablo/rios_FBP_BALANCED_REFERENCED_2026_validation/gt_train/'

#Val images path, use your own path
VAL_IMAGES_PATH = '/home/pablo.canosa/ssd/datasets_pablo/rios_FBP_BALANCED_REFERENCED_2026_validation/img_val/' 
VAL_PNG_MASKS_PATH = '/home/pablo.canosa/ssd/datasets_pablo/rios_FBP_BALANCED_REFERENCED_2026_validation/gt_val/'

#Test images path, use your own path
TEST_IMAGES_PATH = '/home/pablo.canosa/ssd/datasets_pablo/rios_FBP_BALANCED_REFERENCED_2026_validation/img_test/' 
TEST_PNG_MASKS_PATH = '/home/pablo.canosa/ssd/datasets_pablo/rios_FBP_BALANCED_REFERENCED_2026_validation/gt_test/'


try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    SemSegEvaluatorRAWB,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    MaskFormerSemanticDatasetMapperRAWB,
    MaskFormerSemanticDatasetMapperAugmented,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

from detectron2.data import build_detection_train_loader, get_detection_dataset_dicts
from detectron2.data.samplers import RepeatFactorTrainingSampler, TrainingSampler


##################################
# Early Stopping hook
##################################
from detectron2.engine.hooks import HookBase
 
class EarlyStopException(BaseException):
    """Custom exception to elegantly break out of Detectron2's training loop."""
    pass
 
 
class EarlyStoppingHook(HookBase):
    """
    Early stopping with EMA smoothing and minority-class awareness,
    adapted from early_stopping.py for Detectron2's hook system.
 
    Key features
    ------------
    - Composite score: weighted blend of global mIoU and mean IoU of
      minority classes, so the model cannot stop while rare classes are
      still learning.
    - EMA smoothing: the raw composite score is smoothed before each
      comparison to reduce epoch-to-epoch noise on small val sets.
    - Warmup: the patience counter is frozen for the first `warmup_epochs`
      epochs so the pretrained backbone can adapt to 5-band input.
    - Best-weight restoration: on stop, the model's weights are restored
      to the best-seen checkpoint (in-memory copy + saved .pth).
 
    Parameters
    ----------
    patience         : epochs without improvement before stopping.
    min_delta        : minimum improvement to reset the patience counter.
    minority_classes : 0-based class indices considered minority, e.g.
                       [2, 3, 4, 5, 8] for Rock, Asphalt, Concrete, Tiles,
                       Pines in the Galician Rivers dataset.
    minority_weight  : weight of minority mIoU in the composite score
                       (0 = pure global mIoU, 1 = pure minority mIoU).
    ema_alpha        : EMA smoothing factor (higher = less smoothing).
    warmup_epochs    : patience counter is frozen for the first N epochs.
    iters_per_epoch  : needed to convert iteration count to epoch count.
    """
 
    def __init__(
        self,
        patience:         int   = 15,
        min_delta:        float = 0.001,
        minority_classes: list  = None,
        minority_weight:  float = 0.4,
        ema_alpha:        float = 0.3,
        warmup_epochs:    int   = 20,
        iters_per_epoch:  int   = 1,
    ):
        self.patience          = patience
        self.min_delta         = min_delta
        self.minority_classes  = minority_classes or []
        self.minority_weight   = minority_weight
        self.ema_alpha         = ema_alpha
        self.warmup_epochs     = warmup_epochs
        self.iters_per_epoch   = iters_per_epoch
 
        self._ema_score        = None
        self._best_score       = -float("inf")
        self._best_epoch       = 0
        self._best_state       = None   # in-memory deep copy of best state dict
        self._counter          = 0
        self._last_eval_count  = 0      # tracks how many val results we have seen
 
        self.logger = logging.getLogger("detectron2.early_stopping")
 
    # ------------------------------------------------------------------
    # Detectron2 hook entry point
    # ------------------------------------------------------------------
 
    def after_step(self):
        # Only act when a new validation result has been written to storage
        try:
            history = self.trainer.storage.history("sem_seg/mIoU")
        except KeyError:
            return
 
        current_eval_count = len(history.values())
        if current_eval_count <= self._last_eval_count:
            return
        self._last_eval_count = current_eval_count
 
        # Current epoch (1-based)
        epoch = (self.trainer.iter + 1) // self.iters_per_epoch
 
        # --- gather global metrics ---
        val_global = {
            "mIoU":  self._latest("sem_seg/mIoU"),
            "pAcc":  self._latest("sem_seg/pAcc"),
            "mAcc":  self._latest("sem_seg/mAcc"),
            "fwIoU": self._latest("sem_seg/fwIoU"),
        }
 
        # --- gather per-class IoU array ---
        # SemSegEvaluatorRAWB logs per-class IoU as "sem_seg/IoU-<ClassName>"
        # Build the array ordered by class index using ALL_CLASSES defined below
        num_classes = len(ALL_CLASSES)
        per_class_iou = np.zeros(num_classes, dtype=np.float32)
        for cls_idx, cls_name in enumerate(ALL_CLASSES):
            metric_key = f"sem_seg/IoU-{cls_name}"
            per_class_iou[cls_idx] = self._latest(metric_key)
 
        val_per_class = {"iou": per_class_iou}
 
        # --- compute composite score ---
        raw_score = self._composite_score(val_global, val_per_class)
        smoothed  = self._update_ema(raw_score)
 
        # --- check for improvement ---
        is_best = smoothed > self._best_score + self.min_delta
 
        # --- log status ---
        minority_ious = [per_class_iou[c] for c in self.minority_classes]
        self.logger.info(
            f"  [EarlyStopping] epoch={epoch} "
            f"score={smoothed:.4f} (raw={raw_score:.4f})  "
            f"best={self._best_score:.4f} @ ep{self._best_epoch}  "
            f"minority_IoU=[{', '.join(f'{v:.3f}' for v in minority_ious)}]  "
            f"patience={self._counter}/{self.patience}"
            + (" <= warmup" if epoch <= self.warmup_epochs else "")
        )
 
        if is_best:
            self._best_score = smoothed
            self._best_epoch = epoch
            # Save both a .pth checkpoint and an in-memory copy
            self.trainer.checkpointer.save("model_best")
            self._best_state = copy.deepcopy(
                self.trainer.model.state_dict()
            )
            self._counter = 0
            self.logger.info(
                f"  [EarlyStopping] New best score {smoothed:.4f} — "
                f"saved model_best.pth"
            )
        elif epoch > self.warmup_epochs:
            # Only increment patience after warmup
            self._counter += 1
            self.logger.info(
                f"  [EarlyStopping] No improvement. "
                f"Counter: {self._counter}/{self.patience}"
            )
 
            if self._counter >= self.patience:
                self.logger.info(
                    f"  [EarlyStopping] Triggered at epoch {epoch}. "
                    f"Best epoch: {self._best_epoch}  "
                    f"Best score: {self._best_score:.4f}"
                )
                self._restore_best_weights()
                raise EarlyStopException()
 
    # ------------------------------------------------------------------
    # Internal helpers  (mirror early_stopping.py logic)
    # ------------------------------------------------------------------
 
    def _composite_score(self, global_metrics, per_class_metrics):
        """
        Weighted combination of global mIoU and minority-class mIoU.
        If no minority classes are configured, returns plain mIoU.
        """
        miou = global_metrics["mIoU"]
 
        if not self.minority_classes:
            return miou
 
        minority_ious = [per_class_metrics["iou"][c] for c in self.minority_classes]
        # Only count classes that have actually been seen (iou > 0)
        seen = [v for v in minority_ious if v > 0]
        minority_miou = float(np.mean(seen)) if seen else 0.0
 
        return (1.0 - self.minority_weight) * miou + self.minority_weight * minority_miou
 
    def _update_ema(self, raw_score):
        """EMA to smooth out epoch-to-epoch noise in the validation metric."""
        if self._ema_score is None:
            self._ema_score = raw_score
        else:
            self._ema_score = (
                self.ema_alpha * raw_score
                + (1.0 - self.ema_alpha) * self._ema_score
            )
        return self._ema_score
 
    def _restore_best_weights(self):
        """Restore model to the best-seen weights (in-memory copy)."""
        if self._best_state is not None:
            self.trainer.model.load_state_dict(self._best_state)
            self.logger.info(
                f"  [EarlyStopping] Weights restored to epoch {self._best_epoch}."
            )
 
    def _latest(self, metric_name):
        """Return the most recent value for a metric, or 0.0 if not found."""
        try:
            return self.trainer.storage.history(metric_name).values()[-1][0]
        except (KeyError, IndexError):
            return 0.0

##################################
# Early Stopping hook
##################################

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        """
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )"""
        if evaluator_type == "sem_seg_RAWB":
            evaluator_list.append(
                SemSegEvaluatorRAWB(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
    

##################################
# Early Stopping setup in the hooks
##################################
    def build_hooks(self):
        hooks = super().build_hooks()
        
        # Calcular iteraciones por epoch para poder pasárselo al PlottingHook
        import math
        import os
        num_train_images = len(os.listdir(TRAIN_IMAGES_PATH))
        batch_size = self.cfg.SOLVER.IMS_PER_BATCH
        iters_per_epoch = math.ceil(num_train_images / batch_size)
        
        MINORITY_CLASSES = [1, 3, 4, 5, 6, 9]

        # 1. Añadimos Early Stopping
        hooks.append(
            EarlyStoppingHook(
                patience=15,
                min_delta=0.001,
                minority_classes=MINORITY_CLASSES,
                minority_weight=0.4,
                ema_alpha=0.3,
                warmup_epochs=20,
                iters_per_epoch=iters_per_epoch,
            )
        )

        
        # 2. Añadimos nuestro Hook para gráficas
        # hooks.append(
        #     EpochPlottingHook(iters_per_epoch=iters_per_epoch, output_dir=self.cfg.OUTPUT_DIR)
        # )
        
        return hooks
##################################
# Early Stopping setup in the hooks
##################################




#################################
#################################
# Construir el train loader con los repeat factor
#################################
#################################


    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "MaskFormerSemanticDatasetMapperAugmented":
            mapper = MaskFormerSemanticDatasetMapperAugmented(cfg, is_train=True)
        else:
            mapper = MaskFormerSemanticDatasetMapper(cfg, is_train=True)

        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        sampler = None
        if cfg.DATALOADER.SAMPLER_TRAIN == "RepeatFactorTrainingSampler":
            logger = logging.getLogger(__name__)
            logger.info("Building RepeatFactorTrainingSampler for Semantic Segmentation...")
            logger.info(f"Repeat Threshold: {cfg.DATALOADER.REPEAT_THRESHOLD}")

            # Semantic Seg datasets don't have 'annotations' by default. 
            # We must open the masks to see which classes are inside.
            
            logger.info("Scanning mask files to calculate class distribution... (This takes a moment)")
            
            from tqdm import tqdm
            for d in tqdm(dataset_dicts):
                # If 'annotations' is missing, we calculate it from the mask file
                if "annotations" not in d:
                    mask_file = d["sem_seg_file_name"]
                    
                    mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                    
                    if mask is not None:
                        unique_classes = np.unique(mask)
                        
                        # Remove ignore label (e.g. 255)
                        # Ensure you check against the ignore label in your Config
                        ignore_val = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
                        unique_classes = unique_classes[unique_classes != ignore_val]
                        
                        # Create the structure the Sampler expects: [{'category_id': 1}, ...]
                        d["annotations"] = [{"category_id": int(c)} for c in unique_classes]
                    else:
                        d["annotations"] = []

            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD, sqrt=APPLY_SQRT_SMOOTHING_TO_AUGMENTED_DATASET
            )
            
            # Create the actual sampler
            sampler = RepeatFactorTrainingSampler(repeat_factors)

        return build_detection_train_loader(
            cfg, 
            mapper=mapper, 
            sampler=sampler,  # Mandar el sampler
            dataset=dataset_dicts # Mandar el dict con annotations
        )

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

# Load GaoFen Dataset
import cv2
import numpy as np

ALL_CLASSES = [
    "unlabeled",
    "Water",
    "Bare soil",
    "Rock",
    "Asphalt",
    "Concrete",
    "Tiles",
    "Meadows",
    "Native trees",
    "Pines",
    "Eucalyptus"
]

COLOR_LIST = [
    (0, 0, 0),
    (255, 0, 0),
    (124, 72, 7),
    (187, 187, 187),
    (93, 103, 112),
    (255, 225, 25),
    (245, 130, 48),
    (138, 213, 93),
    (60, 180, 75),
    (116, 146, 58),
    (38, 83, 35)

]

ID_TO_COLOR_DICT = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (124, 72, 7),
    3: (187, 187, 187),
    4: (93, 103, 112),
    5: (255, 225, 25),
    6: (245, 130, 48),
    7: (138, 213, 93),
    8: (60, 180, 75),
    9: (116, 146, 58),
    10: (38, 83, 35)
}

def get_gaofen_dict(images_path, gt_dir_png, gt_dir_tif_color): #Creates de dictionary with the information of the dataset in the Detectron2 format, gt_dir_tif_color is not yet used as is for panoptic segmentation
    
    dataset_dicts = []
    number_of_images = len(os.listdir(images_path))
    for image_idx, image_filename in enumerate(os.listdir(images_path)):

        print(f"{image_filename} is image {image_idx+1} out of {number_of_images}")
        record={}

        image_file_path = os.path.join(images_path, image_filename)  


        image_id, _= os.path.splitext(image_filename) # This splits "imagen_patch_0.tif" into "imagen_patch_0" and ".tif"

        # From the image filename, get the id of the GT "imagen_patchgt_0" # Ajustar correctamente para que el nombre coincida
        mask_id = image_id
        gt_mask_grayscale = os.path.join(gt_dir_png, mask_id + ".png")

        record["sem_seg_file_name"] = gt_mask_grayscale

        # Mask size and image size is the same
        height, width = cv2.imread(gt_mask_grayscale, cv2.IMREAD_UNCHANGED).shape[:2]

        record["file_name"] = image_file_path
        record["image_id"] = image_id
        record["height"] = height
        record["width"] = width

        
        record["NIR"] = USE_NIR_BAND 
        record["DATASET_NAME"] = DATASET_NAME

        #End loop and save dict
        dataset_dicts.append(record)
        
        ##########
        #if image_idx == 10:
        #    break
        ##########
    return dataset_dicts

#######
from detectron2.data import MetadataCatalog, DatasetCatalog


def main(args):


    ### Register the datasets
    stuff_dataset_id_to_contiguous_id = {i: i for i in range(11)}#The dictionaries are trivial, the id is the same as the index
    
    # This are the paths for the dataset files in quadrants
    dataset_path_image = TRAIN_IMAGES_PATH
    dataset_path_png_mask = TRAIN_PNG_MASKS_PATH
    dataset_path_tif_mask = '' # Not used yet, for panoptic segmentation

    DatasetCatalog.register("rios_train", lambda : get_gaofen_dict(dataset_path_image,dataset_path_png_mask,dataset_path_tif_mask))
    MetadataCatalog.get("rios_train").stuff_classes = ALL_CLASSES
    MetadataCatalog.get("rios_train").ignore_label = 0
    MetadataCatalog.get("rios_train").thing_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    MetadataCatalog.get("rios_train").stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    
    MetadataCatalog.get("rios_train").stuff_colors = COLOR_LIST
    

    # val trials
    dataset_path_image_val = VAL_IMAGES_PATH
    dataset_path_png_mask_val = VAL_PNG_MASKS_PATH
    dataset_path_tif_mask_val = ''# Not used yet, for panoptic segmentation


    DatasetCatalog.register("rios_val", lambda : get_gaofen_dict(dataset_path_image_val,dataset_path_png_mask_val,dataset_path_tif_mask_val))
    MetadataCatalog.get("rios_val").stuff_classes = ALL_CLASSES 
    MetadataCatalog.get("rios_val").stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    MetadataCatalog.get("rios_val").stuff_colors = COLOR_LIST

    MetadataCatalog.get("rios_val").ignore_label = 0 
    MetadataCatalog.get("rios_val").evaluator_type = "sem_seg_RAWB"



    # test trials
    dataset_path_image_test = TEST_IMAGES_PATH
    dataset_path_png_mask_test = TEST_PNG_MASKS_PATH
    dataset_path_tif_mask_test = ''# Not used yet, for panoptic segmentation


    DatasetCatalog.register("rios_test", lambda : get_gaofen_dict(dataset_path_image_test,dataset_path_png_mask_test,dataset_path_tif_mask_test))
    MetadataCatalog.get("rios_test").stuff_classes = ALL_CLASSES 
    MetadataCatalog.get("rios_test").stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
    MetadataCatalog.get("rios_test").stuff_colors = COLOR_LIST

    MetadataCatalog.get("rios_test").ignore_label = 0 
    MetadataCatalog.get("rios_test").evaluator_type = "sem_seg_RAWB"


    ###
    
    cfg = setup(args)
    # Este codigo sirve para usar max_iters como epochs
    import math
    num_train_images = len(os.listdir(TRAIN_IMAGES_PATH))
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    iters_per_epoch = math.ceil(num_train_images / batch_size)
    
    DESIRED_EPOCHS = cfg.SOLVER.MAX_ITER  # <--- voy a secuestrar el valor original de max_iters
    
    cfg.defrost()
    cfg.SOLVER.MAX_ITER = iters_per_epoch * DESIRED_EPOCHS
    cfg.TEST.EVAL_PERIOD = iters_per_epoch 
    cfg.freeze()
    
    print(f"Info: {num_train_images} imágenes / Batch de {batch_size} = {iters_per_epoch} iteraciones por epoch.")
    print(f"Entrenando por {DESIRED_EPOCHS} epochs (Total: {cfg.SOLVER.MAX_ITER} iteraciones).")
    # Hasta aqui

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    try:
        toret = trainer.train()
    except EarlyStopException:
        print("\n" + "="*50)
        print(" ENTRENAMIENTO DETENIDO POR EARLY STOPPING")
        toret = None # El entrenamiento acabó forzadamente, no hay return de trainer.train()

    print("="*50)
    print("Training completed. Now evaluating on the test set...")
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    cfg.DATASETS.TEST = ("rios_test",) # Cambiamos validación por test
    cfg.freeze()

    best_model = Trainer.build_model(cfg)
    DetectionCheckpointer(best_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )

    res = Trainer.test(cfg, best_model)
    print("Test set evaluation results:")
    print(res)

    return toret




if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
