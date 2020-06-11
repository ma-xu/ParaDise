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
| PD-A-ResNet18 | 70.9861 |89.8457|1.822|11.690|12ms|<a href="">model</a> <a href="">log</a>|
| PD-B-ResNet18 | 72.0873 |90.4177|1.822|11.762|14ms|<a href="">model</a> <a href="">log</a>|
| ResNet50 | 75.8974|92.7224|4.122|25.557|42ms|<a href="">model</a> <a href="">log</a>|
| SE-ResNet50 | 77.2877|93.6478|4.130|28.088|45ms|<a href="">model</a> <a href="">log</a>|
| GE-ResNet50 | 77.1146 |93.7107|4.143|26.06|73ms|<a href="">model</a> <a href="">log</a>|
| AC-ResNet50 |76.5804|93.1820|4.122|25.557|42ms|<a href="">model</a> <a href="">log</a>|
| PD-A-ResNet50 | 76.6867|93.3193|4.122|25.557|42ms|<a href="">model</a> <a href="">log</a>|
| PD-B-ResNet50 |77.3718 |93.4876|4.122|25.636|44ms|<a href="">model</a> <a href="">log</a>|



<br>
<br>
Table 2: Detection performances (%) with different backbones on the MS-COCO validation dataset. We employ two state-of-the-art detectors: RetinaNet and Cascade R-CNN  in our detection experiments.

| Detector | Backbone | AP(50:95) | AP(50) | AP(75) | AP(s)|AP(m)|AP(l)|Download
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Retina|ResNet50|36.2|55.9|38.5|19.4|39.8|48.3|[model](https://drive.google.com/open?id=1imZvUrwg6Vy6TFRLAsL62FsF-DyizZXR) [log](https://drive.google.com/open?id=14rRmHai_9ghL5oC-1DTTiLrt4w_HY0Yl)
|Retina|SE-ResNet50|37.4|57.8|39.8|20.6|40.8|50.3|[model](https://drive.google.com/open?id=1ivzPfC_JhpO7DPs6vzlHGxkZBf2sC60p) [log](https://drive.google.com/open?id=1mKctgPjf9QbEXTeSm_-J_kqeiVNGuMT7)
|Retina|CCD-ResNet50|**37.8**|**58.5**|**40.1**|**21.6**|**41.5**|**50.9**|[model](https://drive.google.com/open?id=1StYpULhwgCwG_ZacBR1bRFqbgt6FRHZr) [log](https://drive.google.com/open?id=1ADWdGj2NcuiK2SCExfWKM8ovypBC68FL)
Cascade R-CNN|ResNet50|40.6|58.9|44.2|22.4|43.7|54.7|[model](https://drive.google.com/open?id=1jGUT2KsFggLSJMkH0cgJUJV_p_cSM-7f) [log](https://drive.google.com/open?id=13g-4XlMlySVUJyrvWeU5FVCA--cojaCk)
Cascade R-CNN|GC-ResNet50|41.1|59.7|44.6|23.6|44.1|54.3|[model](https://drive.google.com/open?id=19cv3TReITDMJuvmAleGzzt3H39iq3pYl) [log](https://drive.google.com/open?id=1uCcKukd4HKtxIc1uUfKydd-_NIPnj9_i)
Cascade R-CNN|CCD-ResNet50|**42.5**|**61.1**|**46.4**|**24.7**|**45.9**|**56.5**|[model](https://drive.google.com/open?id=1655frDSIzUpxjOD4Bt2-l6w0D5DBo2Yn) [log](https://drive.google.com/open?id=1655frDSIzUpxjOD4Bt2-l6w0D5DBo2Yn)
