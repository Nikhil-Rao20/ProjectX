## Datasets
* MS-COCO Person Multi-Task
    * Download images and annotations from [here](https://github.com/ultralytics/ultralytics/pull/5219#issuecomment-1781477032)
    * We would like to thank Andy [@yermandy](https://github.com/yermandy) for providing this dataset.

* CattleEyeView dataset 
    * Download images from https://github.com/AnimalEyeQ/CattleEyeView
    * Multi-task annotations can be found in `./data/CattleEyeView`
* OCHumanApi
    * Download images and annotations from [here](https://cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ochuman)
    * OCHumanApi Github for instructions [here](https://github.com/liruilong940607/OCHumanApi)

* The dataset configuration file can be found in `./config/dataset/cattleeyeview_multitask.yaml` or `./config/dataset/coco_multitask.yaml`.
  * Instructions to modify the configurations can be found in the file.

## Code
* Run the following commands to install mtYOLOv8:
  ```python
  cd ultralytics
  pip install -r requirements.txt
  ```
* The mtYOLOv8 model configuration file and instructions to create other configuration files (e.g., pose, segment, without ECA) can be found in `./config/model/yolov8_multitask_cattleeyeview_ECA.yaml`. 

* The code and instructions to train, validate or predict can be found in `mtYOLO.ipynb`.

* The trained mtYOLOv8 with ECA models for MS-COCO Person Multi-Task and CattleEyeView can be found in `./model_checkpoint`.

#### Conv Block Modifications
To change the Base Convolution Block you can change the name of them to "Conv". Here is a complete list of them.
Go to ProjectX/ultralytics/ultralytics/nn/modules/conv.py

1) Vanilla Block : In line 33, rename the class to "Conv"
2) CBAM Block : In line 140, rename the "CBAM_Conv" to "Conv"
3) ConvNeXt Block : In line 201, rename the "ConvNeXt_Conv" to "Conv"
4) InceptionNeXt Block: In line 263, rename the "InceptionNeXt_Conv" to "Conv"
5) Proposed Block : In line 309, rename the "AdvancedAttentionBasedConvBlock" to "Conv"
**Note : Make sure that while changing the class name to Conv, rename the existing Conv block to their respective name or _Conv. There should be only one Conv class in the file.**

#### Loss Function
We have proposed an gradients based alligned dynamic weighting for the loss. Go to ProjectX/ultralytics/ultralytics/utils/loss.py, at line 802

1) To use the vanilla loss function :

```
self.useBalancer = False
```

2) To use the vanilla loss function :

```
self.useBalancer = True
```

#### EnhancedAttentionResidualBlock : Neck to Head Features enhancement

So we have proposed an another block where we could focus upon very important features for the three outputs from the neck bypassing it through our enhanced attention residual block. 
To use this block you need to go ultralytics/ultralytics/nn/modules/head.py,

1) Vanilla Passing : Comment the lines : 264,265,266,267,293

3) EARB Passing : Un-comment the lines : 264,265,266,267,293

## Acknowledgments
We would like to express our gratitude to 
* [ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 codes
* [@yermandy](https://github.com/yermandy) for the [MS-COCO Person Multi-Task dataset](https://github.com/ultralytics/ultralytics/pull/5219#issuecomment-1781477032) and [multi-task codes](https://github.com/yermandy/ultralytics/tree/multi-task-model) 
* [Efficient Channel Attention by Wang et al. (2020)](https://github.com/BangguWu/ECANet) and [YOLOv8-AM by Chien et al. (2024)](https://github.com/RuiyangJu/Fracture_Detection_Improved_YOLOv8) for the ECA codes
* [Pose2Seg: Detection Free Human Instance Segmentation by Song-Hai et al. (CVPR 2019)](https://github.com/liruilong940607/OCHumanApi) for the OCHumanApi Dataset
