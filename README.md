# 3D-DETR-Caption
This project builds on "End-to-End 3D Dense Captioning with Vote2Cap-DETR" and "Vote2Cap-DETR++." It enhances the model by applying diffusion loss during pretraining to guide global feature extraction and focal loss during training to refine weight allocation.

![1f6c480b9d5974a084ae9a6c18370cbb](https://github.com/user-attachments/assets/22966af0-278a-4fb5-a968-ebde9794b1ba)

## 1. Environment
Our code is tested with PyTorch 2.4.0, CUDA 12.4 and Python 3.8.20.
Besides `pytorch`, this repo also requires the following Python dependencies:

```{bash}
matplotlib
opencv-python
plyfile
trimesh
networkx
scipy
cython
transformers
```

If you wish to use multi-view feature extracted by [Scan2Cap](https://github.com/daveredrum/Scan2Cap), you should also install `h5py`:

```{bash}
pip install h5py
```

It is also **REQUIRED** to compile the CUDA accelerated PointNet++, and compile gIoU support for fast training:

```{bash}
cd third_party/pointnet2
python setup.py install
```

```{bash}
cd utils
python cython_compile.py build_ext --inplace
```

To build support for METEOR metric for evaluating captioning performance, we also installed the `java` package.


## 2. Dataset Preparation

We follow [Scan2Cap](https://github.com/daveredrum/Scan2Cap)'s procedure to prepare datasets under the `./data` folder (`Scan2CAD` **NOT** required).

**Preparing 3D point clouds from ScanNet**. 
Download the [ScanNetV2 dataset](https://github.com/ch3cook-fdu/Vote2Cap-DETR/tree/master/data/scannet) and change the `SCANNET_DIR` to the `scans` folder in [`data/scannet/batch_load_scannet_data.py`](https://github.com/ch3cook-fdu/Vote2Cap-DETR/blob/master/data/scannet/batch_load_scannet_data.py#L16), and run the following commands.

```
cd data/scannet/
python batch_load_scannet_data.py
```

**Preparing Language Annotations**. 
Please follow [this](https://github.com/daveredrum/ScanRefer) to download the ScanRefer dataset, and put it under `./data`.

[Optional] To prepare for Nr3D, it is also required to [download](https://referit3d.github.io/#dataset) and put the Nr3D under `./data`.
Since it's in `.csv` format, it is required to run the following command to process data.

```{bash}
cd data; python parse_nr3d.py
```

## 3. [Optional] Download Pretrained Weights

You can download all the ready-to-use weights from [baidudisk](https://pan.baidu.com/s/1pzVyMepJIE2OC-TRQZrEOw?pwd=gcc8).


## 4. Training and Evaluation

Though we provide training commands from scratch, you can also start with some pretrained parameters provided under the `./pretrained` folder and skip certain steps.

**[optional] 4.0 Pre-Training for Detection**

You are free to **SKIP** the following procedures as they are to generate the pre-trained weights in `./pretrained` folder.

To train the Vote2Cap-DETR's detection branch for point cloud input without additional 2D features (aka [xyz + rgb + normal + height]):

```{bash}
bash scripts/vote2cap-detr++/train_scannet.sh
```

To evaluate the pre-trained detection branch on ScanNet:

```{bash}
bash scripts/vote2cap-detr++/eval_scannet.sh
```

To train with additional 2D features (aka [xyz + multiview + normal + height]) rather than RGB inputs, you can manually replace `--use_color` to `--use_multiview`.


**4.1 MLE Training for 3D Dense Captioning**

Please make sure there are pretrained checkpoints under the `./pretrained` directory. To train the mdoel for 3D dense captioning with MLE training on ScanRefer:

```{bash}
bash scripts/vote2cap-detr++/train_mle_scanrefer.sh
```

And on Nr3D:

```{bash}
bash scripts/vote2cap-detr++/train_mle_nr3d.sh
```

Our MLE training result on ScanRefer can be download form [baidudisk](https://pan.baidu.com/s/1VYPITSDI_jVmjJ6V8D_Mpw?pwd=rd6h).
You can put it under `./exp_scanrefer` folder for evaluating.


**4.2 Evaluating the Weights**

You can evaluate any trained model with specified **models** and **checkponts**. Change `--dataset scene_scanrefer` to `--dataset scene_nr3d` to evaluate the model for the Nr3D dataset.

```{cmd}
bash scripts/eval_3d_dense_caption.sh
```

Run the following commands to store object predictions and captions for each scene.

```{cmd}
bash scripts/demo.sh
```

## 5. Make Predictions for online test benchmark

Our model also provides the inference code for ScanRefer online test benchmark.

The following command will generate a `.json` file under the folder defined by `--checkpoint_dir`.

```
bash submit.sh
```
