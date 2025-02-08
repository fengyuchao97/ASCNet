# ASCNet

## Papers
* Axial-chunked Spatial-temporal Conversation for Change Detection (CVPR 2024) 

The number of F1-Score, FLOPs, and Latency of different CD methods on GZ or SYSU.

![illustration](./images/F1-FLOPs.JPG)

Figure 2: The overall architecture of ASCNet. Regarding the weakly and strongly augmented images, two parallel branches are constructed, i.e., siamese CNN and transformer, to acquire abundant temporal-spatial representations. In practice, we employ GhostNetv2 as the CNN backbone and Video Swim Transformer for the other branch. The former leverages intra-scale axial-chunked interaction for information coupling between bi-temporal image pair, where attention guiding and difference gating are imposed to steer more focus on changes. The latter explores weakly-strongly augmented image sequences, serving to capturing the complementary features that afford strong robustness against interference. Finally, the multi-scale representations encoded by the two branches are progressively decoded through concatenation, upsampling, and residual blocks for change prediction.

![model](./images/1.Model.JPG)

Illustration of intra-scale axial-chunked interaction module, change guiding, and difference gating.

![ITCA](./images/2.Interaction.JPG)

Illustration of axial chunk and squeeze schemes.

![ISE](./images/3.Chunk.JPG)

## 1. Environment setup
This code has been tested on on the workstation with Intel Xeon CPU E5-2690 v4 cores and two GPUs of NVIDIA TITAN V with a single 12G of video memory, Python 3.6, pytorch 1.9, CUDA 10.0, cuDNN 7.6. Please install related libraries before running this code:

    pip install -r requirements.txt

## 2. Download the datesets:
* CLCD-CD:
[CLCD-CD](https://github.com/liumency/CropLand-CD)
* WHU-CD:
[WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
* GZ-CD:
[GZ-CD](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)
* SYSU-CD:
[SYSU-CD](https://github.com/liumency/SYSU-CD)

and put them into `datasets` directory. The directory should be organized as follows: 

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset. 

And change the `root_dir` in the `data_config.py` file.

## 3. Download the models (loading models):

Download the pretrained 'GhostNetv2' model and put it into `pretrained` directory.

And the pretrained models of GSINet on four CD datasets are as follows: 

* [models](https://pan.baidu.com/s/10nL9THYQKQRLnwtv4PZ7fA) code: upu8

and put them into `checkpoints` directory.

## 4. Train
You can find the training script `run_cd.sh` in the folder `scripts`. You can run the script file by `sh scripts/run_cd.sh` in the command environment.

The detailed script file `run_cd.sh` is as follows:

```cmd
gpus=0,1
checkpoint_root=checkpoints 
data_name=GZ  # dataset name 

img_size=256
batch_size=16
lr=0.01
max_epochs=200  #training epochs
net_G=ASCNet # model name
lr_policy=linear

split=train  # training txt
split_val=val  # validation txt
project_name=${net_G}-${data_name}

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}
```

## 5. Evaluate
You can find the evaluation script `eval.sh` in the folder `scripts`. You can run the script file by `sh scripts/eval.sh` in the command environment.

The detailed script file `eval.sh` is as follows:

```cmd
gpus=0
data_name=GZ # dataset name
net_G=ASCNet # model name 
split=test # test.txt
project_name=${net_G}-${data_name} # the name of the subfolder in the checkpoints folder 
checkpoint_name=best_ckpt.pt # the name of evaluated model file 

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
```
    
## 6. Results
Quantitative comparison of CD. The best results are highlighted in RED fonts. All results are described as percentages (\%).

![result](./images/4.Compare.JPG)

Quantitative comparison of CD using metrics for salient detection.

![result2](./images/5.SD.JPG)

Visual results on representative samples, where the left columns are the change predictions and three right columns are the error maps. Each of the two rows in (a)-(h) are samples from CLCD, WHU, GZ, and SYSU, respectively. GT is the ground truth. In predictions, white is a true positive, black is a true negative, red indicates a false positive, and green stands as a false negative. In error maps, yellow is an erroneous detection and blue denotes correct region.

![vis-result](./images/6.Maps.JPG)

Precision and recall values on test samples of WHU dataset.

![vis-WHU](./images/7.FPR.JPG)

## License
Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.
