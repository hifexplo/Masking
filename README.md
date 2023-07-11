# Masking HSI ROIs With Pretrained Models
[HZDR](https://hzdr.de) - [Hif_Exploration](https://www.iexplo.space/)

### **Overview**

The presence of undesired background areas associated with potential noise and unknown spectral characteristics degrades the performance of hyperspectral data processing. Masking out unwanted regions is key to addressing this issue. Processing only regions of interest yields notable improvements in terms of computational costs, required memory, and overall performance.

The proposed method offers a solution for masking out unwanted regions and objects in the images. The method is built upon SAM and Grounding Dino in addition to further computer vision techniques.

To illustrate the efficacy of the masking procedure, the proposed method is deployed on three challenging applications scenarios that demand accurate masking i.e. shredded plastics characterization, drill core scanning, and litter monitoring

![http://url/to/img.png](https://github.com/Elias-Arbash/Masking/blob/main/assets/Methodology.png)

This work builds upon [Segment Anything](https://github.com/facebookresearch/segment-anything) and [Grounding Dino](https://github.com/facebookresearch/segment-anything)

#### **``Requirements``**

Clone and set up the Segment-Anything and Grounding Dino repositories.

Please check the dependency requirements in the two repositories.

The implementation is tested under python 3.8, as well as pytorch 1.12 and torchvision 0.13. We recommend an equivalent or higher pytorch version.

#### **``Input Data``**

For data privacy, the hyperspectral images used in the paper are not fully provided. Only 3 channel representations and their ground truths can be found in the images folder.

For reference, please cite:
