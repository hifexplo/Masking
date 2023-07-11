# Masking HSI ROIs With Pretrained Models
[HZDR](https://hzdr.de) - [Hif_Exploration](https://www.iexplo.space/)

The presence of undesired background areas associated with potential noise and unknown spectral characteristics degrades the performance of hyperspectral data processing. Masking out unwanted regions is key to addressing this issue. Processing only regions of interest yields notable improvements in terms of computational costs, required memory, and overall performance.
The proposed processing pipeline encompasses two fundamental parts: regions of interest mask generation, followed by the application of hyperspectral data processing techniques solely on the newly masked hyperspectral cube. The novelty of our work lies in the methodology adopted for the preliminary image segmentation. We employ the Segment Anything Model (SAM) to extract all objects within the dataset, and subsequently refine the segments with a zero-shot Grounding Dino object detector, followed by intersection and exclusion filtering steps, without the need for fine-tuning or retraining. To illustrate the efficacy of the masking procedure, the proposed method is deployed on three challenging applications scenarios that demand accurate masking i.e. shredded plastics characterization, drill core scanning, and litter monitoring

![http://url/to/img.png](https://github.com/Elias-Arbash/Masking/blob/main/assets/Plastics.png)

This work builds upon [Segment Anything](https://github.com/facebookresearch/segment-anything) and [Grounding Dino](https://github.com/facebookresearch/segment-anything)

**Requirements**

Clone and set up the Segment-Anything and Grounding Dino repositories.

Please check the dependency requirements in the two repositories.

The implementation is tested under python 3.8, as well as pytorch 1.12 and torchvision 0.13. We recommend an equivalent or higher pytorch version.

**Input Data**
For data privacy, the hyperspectral images used in the paper are not fully provided. Only 3 channel representations and their ground truths can be found in the images folder.

For reference, please cite:
