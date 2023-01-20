
This repository contains the codes for the paper *Multimodal Data Augmentation for Image Captioning using Diffusion Models*. 

The codes for the models, training, and evaluations are mainly adapted from [Luo et al.'s work](https://openaccess.thecvf.com/content_cvpr_2018/html/Luo_Discriminability_Objective_for_CVPR_2018_paper.html). You can refer to their [github repository](https://github.com/ruotianluo/ImageCaptioning.pytorch) for more information and instructions. 



## Requirements
- Python 3
- PyTorch 1.3+ (along with torchvision)
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule) (**Remember to follow initialization steps in coco-caption/README.md**)
- yacs
- lmdbdict

## Install


Install submoule *coco-caption* and *cider*

```
git submodule init
git submodule update
```

For *coco-caption*, download materials following the instruction in *coco-caption* repo. If bash doesn't work, you can download the required data manually.



Packages you probably need to install:

```
pip install transformers
pip install opencv-python
pip install gensim
pip install h5py
pip install scikit-image

pip install lmdbdict
pip install yacs
pip install git+https://github.com/ruotianluo/meshed-memory-transformer.git
pip install pyemd
```

Install Java environment.

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
```


data/imagenet_weights/resnet101.pth


If you have difficulty running the training scripts in `tools`. You can try installing this repo as a python package:
```
python -m pip install -e .
```

