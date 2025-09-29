# TFM-PCG-USC
Cuda installation: https://www.cherryservers.com/blog/install-cuda-ubuntu  
Paper: https://arxiv.org/pdf/2112.01527.pdf  
Installation commands under the working folder (cuda 12.4), source: https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md  

Citius Gitlab: https://gitlab.citius.gal/hiperespectral/mask2former-for-multispectral-images    

Env creation:
* conda create --name mask2former python=3.8 -y
* conda activate mask2former

Pytorch and Opencv installation:
* conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
* pip install -U opencv-python

Detectron2 setup:
* git clone git@github.com:facebookresearch/detectron2.git
* cd detectron2
* pip install -e .
* pip install git+https://github.com/cocodataset/panopticapi.git
* pip install git+https://github.com/mcordts/cityscapesScripts.git

Mask2former setup:
* cd ..
* git clone git@github.com:facebookresearch/Mask2Former.git
* cd Mask2Former
* pip install -r requirements.txt
* cd mask2former/modeling/pixel_decoder/ops
* sh make.sh

Working folder:
* cd ~/wip


Nota:

Para el github, en la carpeta donde se hiceron todas las instalaciones, en este caso ~/wip.
* git init
* git remote add origin git@github.com:PCGarcia18/TFM-PCG-USC.git
* git pull origin main



#### SAMPLE
![GIF](samples/transition.gif)


SCRIPTS:
|                    Task                    |                              Script                             |
|:------------------------------------------:|:---------------------------------------------------------------:|
| Semantic on small images                   | train_net_gaofen_semantic_all_classes.py                        |
| Semantic on full size images               | train_net_gaofen_semantic_all_classes_full_size_images.py       |
| Semantic on full size multispectral (WIP)  | train_net_gaofen_semantic_all_classes_full_size_images_hyper.py |
| Panoptic segmentation on small images (WIP)| train_net_gaofen_panoptic.py                                    |
| Semantic on PCA images                     | train_net_gaofen_all_classes_PCA.py                             |



srun python3 train_net_gaofen_semantic_all_classes_full_size_images.py --num-gpus 2 --config-file /home/pablo.canosa/wip/models_and_results/semantic_segmentation_full_size_RGBA/output/config.yaml --eval-only MODEL.WEIGHTS /home/pablo.canosa/wip/models_and_results/semantic_segmentation_full_size_RGBA/output/model_final.pth

srun python3 train_net_gaofen_semantic_all_classes.py --num-gpus 1 --config-file /home/pablo.canosa/wip/Mask2Former/configs/gaofen/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml



# Cambios necesarios para trabajar con 4 o 3 canales en las imagenes RAWB:

* detectron2/detectron2/data/dataset_mapper.py en def __call__(self, dataset_dict): añadir un método para leer las imagenes em,pleadas en test
* Mask2Former/mask2former/data/dataset_mappers/mask_former_semantic_dataset_mapper_RAWB.pyc añadir un nuevo mapper capaz de trabajar con las imagenes RAWB, ya bien sea 4 o 3 canales.
* Mask2Former/train_net_gf_8bit_rawb_small.py obtener un train funcional
* Mask2Former/demo/demo_rawb.py Crear una demo y proporcionar los canales del revés.
* Mask2Former/configs/gaofen/sem-seg Proporcionar una config adecuada.