<div align="center">
  <h1> Object Discovery via Contrastive Learning for Weakly Supervised Object Detection</h1>
</div><div align="center">
  <h3>Jinhwan Seo, Wonho Bae, Danica J. Sutherland, Junhyug Noh, and Daijin Kim</h3>
</div>
<br /><div align="center">
  <img src="./teaser.png" alt="result" width="600"/>
</div>The official implementation of ECCV2022 paper: "Object Discovery via Contrastive Learning for Weakly Supervised Object Detection

## Environment setup:

* [Python 3.7](https://pytorch.org)
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.7.0](https://pytorch.org)
```bash
git clone https://github.com/jinhseo/OD-WSCL/
cd OD-WSCL

conda create --name OD-WSCL python=3.7
conda activate OD-WSCL

pip install ninja yacs cython matplotlib tqdm opencv-python tensorboardX pycocotools
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch

git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../
python setup.py build develop
```
## Dataset:
* [PASCAL VOC (2007, 2012)](http://host.robots.ox.ac.uk/pascal/VOC/)
* [MS-COCO (2014, 2017)](https://cocodataset.org/#download)
Download it and create symlink to datasets/{coco/voc}/
```bash
mkdir -p datasets/{coco/voc}
    datasets/
    ├── voc/
    │   ├── VOC2007
    │   │   ├── Annotations/
    │   │   ├── JPEGImages/
    │   │   ├── ...
    │   ├── VOC2012/
    │   │   ├── ...
    ├── coco/
    │   ├── annotations/
    │   ├── train2014/
    │   ├── val2014/
    │   ├── train2017/
    │   ├── ...
    ├── ...
```
## Proposal:
Download .pkl file from [Dropbox](https://www.dropbox.com/sh/twfxmipuljvyxng/AABomkOHkurFqWAdRpTzrYIAa?dl=0)
```bash
mkdir proposal
    proposal/
    ├── SS/
    │   ├── voc
    │   │   ├── SS-voc07_trainval.pkl/
    │   │   ├── SS-voc07_test.pkl/
    │   │   ├── ...
    ├── MCG/
    │   ├── voc
    │   │   ├── ...
    │   ├── coco
    │   │   ├── MCG-coco_2014_train_boxes.pkl/
    │   │   ├── ...
    ├── ...
```
## Train:
```bash
python -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node={NO_GPU} tools/train_net.py --config-file "configs/{config_file}.yaml" OUTPUT_DIR {output_dir}
```
## Eval:
```bash
python -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node={NO_GPU} tools/test_net.py --config-file "configs/{config_file}.yaml" TEST.IMS_PER_BATCH 8 OUTPUT_DIR {output_dir} MODEL.WEIGHT {model_weight}.pth
```
