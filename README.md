# Paper Title (Yet to be proposed)

This is the official repository for Paper Title 
<br/>**Authors**
<br/>*University*


# Paper
Add up the paper link

## Citation
[Cite accordingly]
<!--

```BibTeX

@INPROCEEDINGS{mtYOLO,
  author={Ong, Kian Eng and Retta, Sivaji and Srinivasan, Ramarajulu and Tan, Shawn and Liu, Jun},
  booktitle={2024 IEEE International Conference on Multimedia and Expo (ICME) Application/Industry}, 
  title={mtYOLO: A multi-task model to concurrently obtain the vital characteristics of individuals or animals}, 
  year={2024},
  volume={},
  number={},
  pages={},
  keywords={},
  doi={}}

```
-->

## Abstract
In multi-task learning, (...continue writing the abstract)
## Model Architecture
[Model architecture design]

## Proposed Block
[Proposing Block architecture design]

## Dynamic Weighting Loss Function
[Equations and mathematical intution of loss function]

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

## Acknowledgments
We would like to express our gratitude to 
* [ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 codes
* [@yermandy](https://github.com/yermandy) for the [MS-COCO Person Multi-Task dataset](https://github.com/ultralytics/ultralytics/pull/5219#issuecomment-1781477032) and [multi-task codes](https://github.com/yermandy/ultralytics/tree/multi-task-model) 
* [Efficient Channel Attention by Wang et al. (2020)](https://github.com/BangguWu/ECANet) and [YOLOv8-AM by Chien et al. (2024)](https://github.com/RuiyangJu/Fracture_Detection_Improved_YOLOv8) for the ECA codes
* [Pose2Seg: Detection Free Human Instance Segmentation by Song-Hai et al. (CVPR 2019)](https://github.com/liruilong940607/OCHumanApi) for the OCHumanApi Dataset
