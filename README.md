# VL-LTR: Learning Class-wise Visual-Linguistic Representation for Long-Tailed Visual Recognition

## Usage

First, install PyTorch 1.7.1+, torchvision 0.8.2+ and other required packages as follows：

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install mmcv==1.3.14
```

## Data preparation

### ImageNet-LT

Download and extract ImageNet train and val images from [here](http://image-net.org/).
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively.

Then download and extract the [wiki text](https://github.com/ChangyaoTian/VL-LTR/releases/download/text-corpus/imagenet.zip) into the same directory, and the directory tree of data is expected to be like this:

```
./data/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
  wiki/
  	desc_1.txt
  ImageNet_LT_test.txt
  ImageNet_LT_train.txt
  ImageNet_LT_val.txt
  labels.txt
```

After that, download the CLIP's pretrained weight `RN50.pt` and `ViT-B-16.pt` into the `pretrained` directory from https://github.com/openai/CLIP.

### Places-LT

Download the `places365_standard` data from [here](http://places2.csail.mit.edu/download.html). (For researchers in Chinese Mainland, you can download the long-tailed version directly from Baidu Netdisk [6x8u](https://pan.baidu.com/s/11rOZgolQwJlygZNq2XN9Fw).)

Then download and extract the [wiki text](https://github.com/ChangyaoTian/VL-LTR/releases/download/text-corpus/places.zip) into the same directory. The directory tree of data is expected to be like this (almost the same as ImageNet-LT):

```
./data/places/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
  wiki/
  	desc_1.txt
  Places_LT_test.txt
  Places_LT_train.txt
  Places_LT_val.txt
  labels.txt
```

### iNaturalist 2018

Download the `iNaturalist 2018` data from [here](https://github.com/visipedia/inat_comp/tree/master/2018).

Then download and extract the [wiki text](https://github.com/ChangyaoTian/VL-LTR/releases/download/text-corpus/iNat.zip) into the same directory. The directory tree of data is expected to be like this:

```
./data/iNat/
  train_val2018/
  wiki/
  	desc_1.txt
  categories.json
  test2018.json
  train2018.json
  val2018.json
```

## Evaluation

To evaluate VL-LTR with a single GPU run:

- Pre-training stage

```sh
bash eval.sh ${CONFIG_PATH} 1 --eval-pretrain
```

- Fine-tuning stage:

```sh
bash eval.sh ${CONFIG_PATH} 1
```

The `${CONFIG_PATH}` is the relative path of the corresponding configuration file in the `config` directory.

## Training

To train VL-LTR on a single node with 8 GPUs for:

- Pre-training stage, run:

```sh
bash dist_train_arun.sh ${PARTITION} ${CONFIG_PATH} 8
```

- Fine-tuning stage:

  - First, calculate the $\mathcal L_{\text{lin}}$ of each sentence for AnSS method by running this:

  ```sh
  bash eval.sh ${CONFIG_PATH} 1 --eval-pretrain --select
  ```

  - then, running this:

  ```sh
  bash dist_train_arun.sh ${PARTITION} ${CONFIG_PATH} 8
  ```

The `${CONFIG_PATH}` is the relative path of the corresponding configuration file in the `config` directory.

## Results

Below list our model's performance on ImageNet-LT, Places-LT, and iNaturalist 2018.

|     Dataset     |  Backbone  |   Top-1 Accuracy |  Download |
| :--------------: | :---------: | :------------: | :------: |
|   ImageNet-LT   |  ResNet-50  |   70.1   | [Weights](https://github.com/ChangyaoTian/VL-LTR/releases/download/checkpoints/imageNet-LT_r50.zip) |
|   ImageNet-LT   | ViT-Base-16 |  77.2  | [Weights](https://github.com/ChangyaoTian/VL-LTR/releases/download/checkpoints/imageNet-LT_vit16.zip) |
|    Places-LT    |  ResNet-50  |   48.0 | [Weights](https://github.com/ChangyaoTian/VL-LTR/releases/download/checkpoints/places_r50.zip) |
|    Places-LT    | ViT-Base-16 |    50.1 | [Weights](https://github.com/ChangyaoTian/VL-LTR/releases/download/checkpoints/places_vit16.zip)    |
| iNaturalist 2018 |  ResNet-50  |  74.6  | [Weights](https://github.com/ChangyaoTian/VL-LTR/releases/download/checkpoints/inat_finetune_r50.zip)  |
| iNaturalist 2018 | ViT-Base-16 | 76.8   | [Weights](https://github.com/ChangyaoTian/VL-LTR/releases/download/checkpoints/inat_finetune_vit16.zip) |

For more detailed information, please refer to our [paper](https://arxiv.org/abs/2111.13579) directly.

## Citation

If you are interested in our work, please cite as follows:

```
@article{tian2021vl,
  title={VL-LTR: Learning Class-wise Visual-Linguistic Representation for Long-Tailed Visual Recognition},
  author={Tian, Changyao and Wang, Wenhai and Zhu, Xizhou and Wang, Xiaogang and Dai, Jifeng and Qiao, Yu},
  journal={arXiv preprint arXiv:2111.13579},
  year={2021}
}
```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
