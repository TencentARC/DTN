# Dynamic Token Normalization Improves Vision Transformers

This is the PyTorch implementation of the paper **Dynamic Token Normalization Improves Vision Transfromers**. Codea and Models will be available soon.


## Dynamic Token Normalization
We design a novel normalization method, termed Dynamic Token Normalization (DTN), which inherits the advantages from LayerNorm and InstanceNorm. DTN can be seamlessly plugged into various transformer models, consistenly improving the performance.
<div align=center><img src="DTN.png" width="1080" height="250"></div>

**Comparisons of top-1 accuracies** on the validation set of ImageNet, by using ViT trained with LN and DTN.

|Model|Top-1|Top-5|
| :----:  | :--: |:--: |
| ViT-T*-LN | 72.3|91.4|
| ViT-T*-DTN | 73.2|91.7|
| ViT-S*-LN | 80.6|95.2|
| ViT-S*-DTN | 81.7|95.8|
| ViT-B*-LN | 81.7|95.8|
| ViT-B*-DTN | 82.5|96.1|

## Getting Started
* Install [PyTorch](http://pytorch.org/)
* Clone the repo:
  ```
  git clone https://github.com/dtn-anonymous/DTN.git
  ```

### Requirements

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.3.2`:

```bash
pip install timm==0.3.2
```

### Data Preparation
- Download the ImageNet dataset which should contain train and val directionary and the txt file for correspondings between images and labels.

### Training a model from scratch
An example to train our DTN is given in DTN/scripts/train.sh. To train ViT-S* with our DTN, 
```
cd DTN/scripts   
sh train.sh layer vit_norm_s_star configs/ViT/vit.yaml
```
Number of GPUs and configuration file to use can be modified in train.sh







