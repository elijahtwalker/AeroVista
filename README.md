# AeroVista: Leveraging Computer Vision to Create High-Performance Search and Rescue Drones

## Contributors
* **Research Lead:** [Elijah Walker](https://www.linkedin.com/in/elijahtruthwalker/)
* **Research Team:** [Azia Bay-Asen](https://www.linkedin.com/in/aziabay/), [Ibrahim Abdel Rahman](https://www.linkedin.com/in/ibrahim-abdel-rahman/), [Aba Onumah](https://www.linkedin.com/in/aba-onumah-63315328b/), [Adrian Tran](https://www.linkedin.com/in/adrianvtran/), [Sreevasan Sivasubramanian](https://www.linkedin.com/in/sreevasan-sivasubramanian-0a3844228/)
* **Faculty Advisors:** Dr. Yapeng Tian, Dr. Yu Xiang

## Poster

![AeroVista Research Poster](https://github.com/IbrahimARahman/AeroVista/assets/99378816/d39eca17-0fba-4d32-a92c-22a18667ea66)

**Tags:** Artificial Intelligence (AI), Machine Learning (ML), Computer Vision (CV), Object Detection, Object Classification, Instance Segmentation, Image Segmentation, Real-Time Object Detector (RTMDet), Mask Region-Based Convolutional Neural Network (Mask R-CNN), Python, PyTorch, Torchvision, Albumentations, Image Augmentation, Voxel51, Search and Rescue (SAR), Search and Rescue Image Dataset for Person Detection (SARD), Unmanned Aerial Vehicles (UAVs), DJI Tello drones, DJITelloPy

## Introduction

Search and Rescue (SAR) ...

## Motivation

We sought to examine ...

## Dataset and Experimental Setup

Our dataset ...

## Architectures

### Mask R-CNN

The mask ...

#### Backbone

The backbone consists of ...

#### Inner Layers

The inner processing ...

#### The Head

In contrast to ...

### RTMDet

The Real-Time ...

#### Backbone

The backbone consists of ...

#### Neck

RTMDet uses the Feature Pyramid Attention Module with Positional Encoding for Object Detection (PAFPN) for its neck architecture, fusing multi-level features from the backbone. PAFPN is built on Feature Pyramid Networks (FPNs), popular neural network architectures used for object detection. FPNs come with the disadvantage of having long information paths between lower layers and top-most features. PAFPN overcomes this disadvantage by combining FPNs with bottom-up path augmentation. The neck architecture uses the same basic building blocks as the backbone but includes bottom-up and top-down feature propagation to enhance the overarching pyramid feature map.

#### Head

In contrast to ...

## Results

As shown in the figures ...

## Analysis

### Appropriate Fit

The Mask R-CNN loss functions for training and validation in Figure 7 maintain relative proximity and converge near the same horizontal line. If the functions did not converge, it would be a sign of either overfitting or underfitting, or that the model has failed to achieve a balance between learning from the training data and generalizing to new data. Instead, the graphed results for Mask R-CNN demonstrate appropriate fit.

### Continued Learning

Both the Mask R-CNN loss functions for training and validation in Figure 7 and the RTMDet loss functions for training in Figure 8 gradually decrease, signifying that both models continued to learn from the training data over time.

### Epochs

Notably, RTMDet takes more epochs to converge when compared to Mask R-CNN. To combat this difference, we trained RTMDet over forty epochs and Mask R-CNN over ten.

### Mean Average Precision (mAP)

The table under Figure 5 and Figure 6 exhibits the box mAP50 and mask mAP50 values we obtained for Mask R-CNN and RTMDet. Our results are higher when compared to the corresponding metrics for both base models, demonstrating the effectiveness of our model fine-tuning methods and dataset training.

### Intersection over Union (IoU)

We chose to compare mAP values at IoU 0.5. We achieved the greatest mAP results at the threshold, and it is sufficient for the task of object detection.

## Conclusion

In summary ...

## References

1. Wu, Minghu et al. "Object detection based on RGC mask R-CNN." IET Image Processing, 13 May 2020, 14: 1502-1508. [https://doi.org/10.1049/iet-ipr.2019.0057](https://doi.org/10.1049/iet-ipr.2019.0057).
2. Lyu, Chengqi et al. "RTMDet: An Empirical Study of Designing Real-Time Object Detectors." arXiv, 16 Dec. 2022. [https://arxiv.org/abs/2212.07784](https://arxiv.org/abs/2212.07784).
3. Shastry, Animesh. "SARD_YOLO Dataset." Roboflow Universe, Feb. 2022. [https://universe.roboflow.com/animesh-shastry/sard_yolo](https://universe.roboflow.com/animesh-shastry/sard_yolo).
4. He, Kaiming et al. "Mask R-CNN." arXiv, 24 Jan. 2018. [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870).
5. "SARSAT U.S. Rescues Map." Department of Commerce: SARSAT, 26 Apr. 2024. [https://www.sarsat.noaa.gov/sarsat-us-rescues/](https://www.sarsat.noaa.gov/sarsat-us-rescues/).
