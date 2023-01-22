# Multimodal Data Augmentation for Image Captioning using Diffusion Models

This repository contains the codes for the paper *Multimodal Data Augmentation for Image Captioning using Diffusion Models*. 

The codes for the models, training, and evaluations are mainly adapted from [Luo et al.'s work](https://openaccess.thecvf.com/content_cvpr_2018/html/Luo_Discriminability_Objective_for_CVPR_2018_paper.html). You can refer to their [github repository](https://github.com/ruotianluo/ImageCaptioning.pytorch) for more information and instructions. 





# Install

## Feature Extraction

For more detailed information, please refer to the README in `data/`. 

### FC Features

Pretrained resnet models (e.g., resnet101) should be downloaded and should placed in `data/imagenet_weights`. 

### Bottom-Up Features

We use a *PyTorch* implementation of Bottom-Up attention instead of the original *caffe* one. Follow the instructions in [Bottom-Up attention repository](https://github.com/MILVLG/bottom-up-attention.pytorch) to install. 

Requirements: 
- Python >= 3.6
- PyTorch >= 1.4
- Cuda >= 9.2 and cuDNN
- Apex
- Detectron2
- Ray
- OpenCV
- Pycocotools


## Evaluation

Requirements: 
- Python 3
- PyTorch 1.3+ (along with torchvision)
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule) (**Remember to follow initialization steps in coco-caption/README.md**)

Install submoule *coco-caption* and *cider*

```
git submodule init
git submodule update
```

For *coco-caption*, download materials following the instruction in [*coco-caption* repository](https://github.com/ruotianluo/coco-caption/tree/ea20010419a955fed9882f9dcc53f2dc1ac65092). If bash doesn't work, you can download the required data manually.

Packages you may need to install:

```
pip install transformers
pip install opencv-python
pip install gensim
pip install h5py
pip install scikit-image

pip install lmdbdict
pip install yacs
pip install pyemd
```

Note that Java environment should be installed for COCO evaluation. 

<!-- Install Java environment.

```
tar -zxvf jdk-8u221-linux-x64.tar.gz
```

```
vim /etc/profile
```

Enter `i` to edit

```
export JAVA_HOME=/root/jdk1.8.0_221
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib:$CLASSPATH
export JAVA_PATH=${JAVA_HOME}/bin:${JRE_HOME}/bin
export PATH=$PATH:${JAVA_PATH}
```

`esc` and enter `:wq` to save the edited profile. Input to activate

```
source /etc/profile
``` -->


## Data Quality Assessment

The implementation of the three multimodal data quality assessment methods mentioned in the paper: 

- MUSIQ: [PyTorch Toolbox for Image Quality Assessment](https://github.com/chaofengc/IQA-PyTorch).

- CLIPScore: [Codes for EMNLP 2021 paper CLIPScore](https://github.com/jmhessel/clipscore). 

- VIFIDEL: Codes adapted from the original [VIFIDEL codes](https://github.com/ImperialNLP/vifidel). 

For more details, please refer to `quality/`. 



---

If you have difficulty running the training scripts in `tools`. You can try installing this repo as a python package:
```
python -m pip install -e .
```


# Basic Usage

1. Construct synthetic datasets
2. Extract features for FC model and Transformer-based models
3. Train FC and Transformer-based models
4. (Multimodal data quality assessment and data selection)
5. Evaluation

## Text-to-Image Synthesis and Synthetic Datasets Construction

You can follow the instructions in [*Stable Diffusion*](https://github.com/CompVis/stable-diffusion) or tutorials anywhere else. 

An example code that we used for text-to-image synthesis is in `sd/`. The synthetic datasets in the format of *coco datasets* are also provided. However, we are not allowed by the anonymous policy to offer explicit Google Drive links so far, and the synthetic images are too large (~20G) to upload directly in the paper submission page. Nonetheless, **the access to the whole synthetic dataset will be provided immediately after the anonymous period**. 

## Feature Extraction

After successful installation, you can run the following example commands to extract features for the captioning models. Note that the file dir in the command lines should be replaced depending on your situation. 

Data pre-processing:

```
python3 scripts/prepro_labels.py \
        --input_json /data/sd_aug_dataset.json \
        --output_json /output_dir/sd.json \
        --output_h5 /output_dir/sd \
        --use_coco_vocab 0
```

FC features:

```
python3 scripts/prepro_feats.py \
        --input_json /data/sd_aug_dataset.json \
        --output_dir /output_dir \
        --images_root /image_dir
```

Bottom-Up features: 

```
python3 extract_features.py --mode caffe \
         --num-cpus 32 --gpus '0' \
         --extract-mode roi_feats \
         --min-max-boxes '10, 100' \
         --config-file configs/caffe/test-caffe-r101.yaml \
         --image-dir /image_dir --bbox-dir /output_bbox_dir \
         --out-dir /output_bu_feat_dir
```

## Model Training and Evaluation

The model training codes are the same as those in [Luo et al.'s repository](https://github.com/ruotianluo/ImageCaptioning.pytorch). 

A simplified version is using *yml* files. The configurations used in the paper are in `configs/`. 

```
python3 tools/train.py --cfg configs/fc/fc.yml --id fc --vis_device 0
```

The evaluation command: 

```
python3 tools/eval.py --dump_images 0 --num_images 5000 \
        --model /logs/log_fc_sd/model-best.pth \
        --infos_path /logs/log_fc_sd/infos_fc_sd-best.pkl \
        --input_json /data/sd.json \
        --input_label_h5 /data/sd_label.h5 \
        --language_eval 1 --force 1 
```

## Data Quality Assessment

For more details about the implementation of MUSIQ, CLIPScore, and VIFIDEL, please refer to `quality/`. 


# Examples

Datasets used in the paper are not included in this repo because of their large size. We will offer Google Drive links after the anonymous period. 

We will make our synthetic images public later, and know we offer some exmples in `examples/`. *365525_coco.jpg* stands for the COCO image and *365525_sd.jpg* for the corresponding synthetic image generated by *Stable Diffusion*. 



