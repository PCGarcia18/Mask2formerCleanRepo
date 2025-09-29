# Las imágenes están en formato tiff, son de 16 bits, 112*112 y tienen 4 bandas. Las bandas están en formato NIR R G B. Al estar en un formato estándar se pueden cargar directamente a un dataset de HuggingFace.

from datasets import load_dataset, Image
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, SwinConfig
from evaluate import load
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


################################################
################################################
# Load dataset
################################################
################################################


tif_dataset = "/home/pablo.canosa/gaofen_clasificacion/pruebas_pablo/tif_dataset"

dataset = load_dataset(
    "imagefolder",
    data_dir=tif_dataset,
)

# Optional: Cast the image column to avoid automatic decoding
dataset = dataset.cast_column("image", Image(decode=False))


def read_raw_data(example):
    path = example['image']['path']  

    dataset = gdal.Open(path)
    img = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount))
    for i in range(dataset.RasterCount):
        img[:, :, i] = dataset.GetRasterBand(i + 1).ReadAsArray()
    
    example['raw_data'] = img
    return example

# Apply the transformation to the dataset
dataset = dataset.map(read_raw_data)




image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
mean=image_processor.image_mean
std=image_processor.image_std

# esto lo añadí a macheta para que funcione con 4 canales
mean = [0.5, mean[0],mean[1],mean[2]] #NIR R G B
std = [0.25, std[0],std[1],std[2]] #NIR R G B
mod_image_processor = image_processor
mod_image_processor.image_std = std
mod_image_processor.image_mean = mean
print(mod_image_processor)


################################################
################################################
# Load model directly
################################################
################################################

metric = load("accuracy")

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

print(label2id)


normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")




# BEFORE I CALL IT I NEED TO MAKE SURE THAT IT IS A NUMPY ARRAY
train_transforms = Compose(
        [
            ToTensor(),
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            ToTensor(),
            Resize(size),
            CenterCrop(crop_size),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    #"""Apply train_transforms across a batch."""
    #example_batch["pixel_values"] = [
    #    train_transforms(image.convert("RGB")) for image in example_batch["image"]
    #]
    """
    example_batch["pixel_values"] = []
    for image in example_batch["raw_data"]:
        array = np.array(image)
        # cast it to iunt8
        array = array.astype(np.uint8)
        array = train_transforms(array)
        example_batch["pixel_values"].append(array)
    """
    example_batch["pixel_values"] = [train_transforms(np.array(image).astype(np.float32)) for image in example_batch["raw_data"]]

    return example_batch

def preprocess_val(example_batch):
    #"""Apply val_transforms across a batch."""
    #example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    
    """
    example_batch["pixel_values"] = []
    for image in example_batch["raw_data"]:
        array = np.array(image)
        # cast it to iunt8
        array = array.astype(np.uint8)
        array = val_transforms(array)
        example_batch["pixel_values"].append(array)
        print(example_batch)
    """
    example_batch["pixel_values"] = [val_transforms(np.array(image).astype(np.float32)) for image in example_batch["raw_data"]]

    return example_batch



splits = dataset["train"].train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

print(splits)
print(train_ds)
print(val_ds)
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)


################################################
################################################
# Parameters
################################################
################################################

config = SwinConfig.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
config.embed_dim = 96  # change embedding dimension to mask2former value
config.num_heads = [3, 6, 12, 24]  # change number of heads for each stage to mask2former value
config.depths = [2, 2, 6, 2]  # change depths for each stage to mask2former value
config.window_size = 7  # change window size to mask2former value
config.ape = False  # change absolute position embedding to mask2former value
config.drop_path_rate = 0.3  # change drop path rate to mask2former value
config.patch_norm = True  # change patch norm to mask2former value
config.id2label = id2label # add the labels of the dataset eurosat, we have 24 classes instead of 1000
config.label2id = label2id # add the labels of the dataset eurosat, we have 24 classes instead of 1000
config.num_channels = 4 # If I can get the multispectral images to work, I will have to change this value so that it uses the NIR value
batch_size = 24


model_checkpoint = "microsoft/swin-tiny-patch4-window7-224"

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    config=config,
    #label2id=label2id,
    #id2label=id2label,
    ignore_mismatched_sizes=True,
)


model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-GAOFEN-MULTI-Intento2",
    remove_unused_columns=False,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size = batch_size,
    gradient_accumulation_steps = 4,
    per_device_eval_batch_size = batch_size,
    num_train_epochs=12,# en el tuto pone 3 pero tarda mucho
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)




# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)



# define a collate_fn, which will be used to batch examples together. Each batch consists of 2 keys, namely pixel_values and labels.
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)


train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate()
# some nice to haves:
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)








