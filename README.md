# ParaDise: Parameter Disentanglement for Neural Networks
(Underview by NeurIPS 2020)
<br><br>

## Requirements
PyTorch>=1.3.0<br>
NVIDIA/Apex<br>
NVIDIA/DALI<br>

## Introduction
In this project, we revisit the learnable parameters in neural networks, and prove that it is feasible to disentangle learnable parameters to latent sub-parameters, which focus on different patterns and representations, to enhance the learning capacity of a network. This important finding leads us to study further the aggregation of discriminative representations in one layer. We design the  parameter disentanglement (ParaDise), which trains a network by considering diverse patterns in parallel, and aggregates them into one for inference. Using ParaDise, we significantly improve the learning capacity of a network while maintaining the same complexity for inference. To further enhance the discriminative representations, we develop a highly light-weight refinement module, which adaptively refines the combination of diverse representations according to the input. Theories of overparameterization and lottery tickets hypothesis verify the effectiveness of our method. 



## Implementation
In this repository, all the models are implemented by [pytorch](https://pytorch.org/).<br>

We use the standard data augmentation strategies with [ResNet](https://github.com/pytorch/examples/blob/master/imagenet/main.py).<br>

:blush: `All trained models and training log files are submitted to Google Drive.`

:blush: `We provide corresponding links in the "download"  column.`

You can use the following commands to test a dataset.

```shell
git clone ParaDise
cd ParaDise
# change 8 to your GPU number, '--fp16' indicates half precision for fast training. '--b' batch size.
# for more configures, see imagenet.py.
python3 -m torch.distributed.launch --nproc_per_node=8 imagenet.py -a pd_a_resnet18 --fp16 --b 32
```


## ImageNet classification
<br>
<br>
Table:  Comparison results of single-crop classification accuracy (%) and complexity on the ImageNet validation set.

| Model | top-1 acc. |top-5 acc. |FLOPs(G)|Parameters(M)|Latency(cpu)|Download|
| --- | --- |--- |--- |--- |---|---|
| ResNet18 | 69.6349 |89.0047|1.822|11.690|12ms|<a href="">model</a> <a href="">log</a>|
| SE-ResNet18 | 71.0236 |89.9159|1.823|11.779|13ms|<a href="">model</a> <a href="">log</a>|
| GE-ResNet18 | 70.4046 |89.7780|1.825|11.753|16ms|<a href="">model</a> <a href="">log</a>|
| AC-ResNet18 | 70.7789 |89.6763|1.822|11.690|12ms|<a href="">model</a> <a href="">log</a>|
| PD-A-ResNet18 | 70.9861 |89.8457|1.822|11.690|**12ms**|<a href="">model</a> <a href="">log</a>|
| PD-B-ResNet18 | **72.0873** |**90.4177**|1.822|11.762|**14ms**|<a href="">model</a> <a href="">log</a>|
| ResNet50 | 75.8974|92.7224|4.122|25.557|42ms|<a href="">model</a> <a href="">log</a>|
| SE-ResNet50 | 77.2877|93.6478|4.130|28.088|45ms|<a href="">model</a> <a href="">log</a>|
| GE-ResNet50 | 77.1146 |**93.7107**|4.143|26.06|73ms|<a href="">model</a> <a href="">log</a>|
| AC-ResNet50 |76.5804|93.1820|4.122|25.557|42ms|<a href="">model</a> <a href="">log</a>|
| PD-A-ResNet50 | 76.6867|93.3193|4.122|25.557|**42ms**|<a href="">model</a> <a href="">log</a>|
| PD-B-ResNet50 |**77.3718** |93.4876|4.122|25.636|**44ms**|<a href="">model</a> <a href="">log</a>|



<br>
<br>
Table: Ablation studies of the branches based on ResNet18.

| Standard1 | Standard2 |Group|Skeleton|PD-A|PD-A|PD-A|PD-B|PD-B|PD-B|
| --- | --- |--- |--- |--- |---|---|---|---|---|
|  |  | | |top-1|top5|Download|top1|top-5|Download|
|  :heavy_check_mark:| | | |69.6349|89.0047|<a href="">model</a> <a href="">log</a>|-|-|-|
|  :heavy_check_mark:| :heavy_check_mark: | | |70.9881|89.8218|<a href="">model</a> <a href="">log</a>|71.8990|90.3739|<a href="">model</a> <a href="">log</a>|
|  :heavy_check_mark:| |  :heavy_check_mark:| |70.1830|89.4133|<a href="">model</a> <a href="">log</a>|70.0474|89.3156|<a href="">model</a> <a href="">log</a>|
|  :heavy_check_mark:| |  |:heavy_check_mark: |70.7789|89.6763|<a href="">model</a> <a href="">log</a>|71.9872|90.4157|<a href="">model</a> <a href="">log</a>|
|  :heavy_check_mark:| | :heavy_check_mark: |:heavy_check_mark: |**71.1799**|89.8278|<a href="">model</a> <a href="">log</a>|71.8232|90.2524|<a href="">model</a> <a href="">log</a>|
|  :heavy_check_mark:| :heavy_check_mark:| :heavy_check_mark: |:heavy_check_mark: |70.9861|**89.8457**|<a href="">model</a> <a href="">log</a>|**72.0873**|**90.4177**|<a href="">model</a> <a href="">log</a>|




## Object Detection on MS COCO benchmark
We employ the [mmdetection](https://github.com/open-mmlab/mmdetection) framework for our object detection task. The only required operation is replacing the backbone to our ParDise variants.

- **TO DO:** applying ParaDise to detectors, not only the backbone models.

Table: Detection performance on MS COCO benchmark.

| Detector | Backbone | AP(50:95) | AP(50) | AP(75) | AP(s)|AP(m)|AP(l)|Download
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Retina|ResNet50|36.2|55.9|38.5|19.4|39.8|48.3|[model](https://drive.google.com/open?id=1imZvUrwg6Vy6TFRLAsL62FsF-DyizZXR) [log](https://drive.google.com/open?id=14rRmHai_9ghL5oC-1DTTiLrt4w_HY0Yl)
|Retina|PD-A-ResNet50|36.8|56.9|39.3|20.2|40.7|49.4|[model]() [log]()
|Retina|PD-B-ResNet50|37.9|58.6|40.1|21.3|40.8|50.7|[model]() [log]()
Cascade R-CNN|ResNet50|40.6|58.9|44.2|22.4|43.7|54.7|[model](https://drive.google.com/open?id=1jGUT2KsFggLSJMkH0cgJUJV_p_cSM-7f) [log](https://drive.google.com/open?id=13g-4XlMlySVUJyrvWeU5FVCA--cojaCk)
Cascade R-CNN|PD-A-ResNet50|41.7|60.4|45.3|23.7|44.5|55.3|[model]() [log]()
Cascade R-CNN|PD-B-ResNet50|42.1|61.0|45.7|24.3|45.3 |55.5|[model]() [log]()

## Other visual tasks

We argure that our ParaDise is suitable for other visual tasks, like **segmentation**, **keypoints detection**, etc. More expriments on other tasks are ongoing. 

