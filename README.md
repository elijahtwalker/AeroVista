# AeroVista: Leveraging Computer Vision to Create High-Performance Search and Rescue Drones

## Contributors
* **Research Lead:** [Elijah Walker](https://www.linkedin.com/in/elijahtruthwalker/)
* **Research Team:** [Azia Bay-Asen](https://www.linkedin.com/in/aziabay/), [Ibrahim Abdel Rahman](https://www.linkedin.com/in/ibrahim-abdel-rahman/), [Aba Onumah](https://www.linkedin.com/in/aba-onumah-63315328b/), [Adrian Tran](https://www.linkedin.com/in/adrianvtran/), [Sreevasan Sivasubramanian](https://www.linkedin.com/in/sreevasan-sivasubramanian-0a3844228/)
* **Faculty Advisors:** Dr. Yapeng Tian, Dr. Yu Xiang

## Poster

![AeroVista Research Poster](https://github.com/IbrahimARahman/AeroVista/assets/99378816/d39eca17-0fba-4d32-a92c-22a18667ea66)

**Tags:** Artificial Intelligence (AI), Machine Learning (ML), Computer Vision (CV), Object Detection, Object Classification, Instance Segmentation, Image Segmentation, Real-Time Object Detector (RTMDet), Mask Region-Based Convolutional Neural Network (Mask R-CNN), Python, PyTorch, Torchvision, Albumentations, Image Augmentation, Voxel51, Search and Rescue (SAR), Search and Rescue Image Dataset for Person Detection (SARD), Unmanned Aerial Vehicles (UAVs), DJI Tello drones, DJITelloPy

## Introduction

Search and rescue (SAR) operations are vital for ensuring the safety of people in times of danger. Since 1982, over 50,000 people have been rescued worldwide from these SAR missions, which proves their efficacy in saving human life. These efforts can be streamlined using autonomous devices, with one of the most efficient being drones. Their portability, relatively low cost, and speed make them one of the best options in these types of operations. They are also scalable, since many autonomous drones can work in parallel without any human input. Recently, target pinpointing has been enhanced using machine learning, which are commonly Convolution Neural‐Networks, or CNNs. These innovative models can be used to benefit SAR mission strategy through image segmentation.

## Motivation
Following what Ibrahim said, we sought to examine the performance of autonomous drones with image detection models and determine the most efficient and reliable model to utilize in search and rescue drones. 
For our research, we used Mask-RCNN and RTMDET-Ins as the models for testing. We trained both these models on a search and rescue dataset containing images and also tested the model with custom images made by our group. We hope that via our research, we can find an efficient model example that can be used in actual implementations of search and rescue drones to help save individuals in distress. 

## Dataset and Experimental Setup

<img width="932" alt="SARDYOLODATASET" src="https://github.com/IbrahimARahman/AeroVista/assets/108421238/bea93ae2-6a2f-414f-90c4-c9b1e7ac65b0">


Our dataset is the SARD_YOLO dataset, which includes almost 2,000 overhead images of people in various environments. Many of these photos portray people in distress or hard-to-find locations, which is critical for search-and-rescue operations. For our validation set, we captured 100 images of our own using a Tello drone, featuring similar images that were manually annotated with Voxel51. The purpose of the validation set is to verify the efficacy of the model outside of the SARD_YOLO dataset, as well as testing its proficiency under a variety of real-world conditions. 

These conditions, such as fog, rain, or lighting differences, can impair the clarity of a drone's camera and the accuracy of the model. In order to account for these, we augmented the SARD_YOLO dataset for training. Using transformations like gaussian blurs, color jitters, elastic transformations, and solarization, we greatly enhanced the model's flexibility and accuracy. Some of these transformations, such as gaussian blurs, emulated foggy or rainy weather. In addition to augmenting the dataset, our validation set included many images at different angles, captured using a 3D-printed mirror mounted to the drone. 

<div style="justify-content: center;">
    <img width="300" alt="Drone2" src="https://github.com/IbrahimARahman/AeroVista/assets/108421238/4a33317c-4bab-4aef-9431-f42f4b16eb2c">
    <img width="400" alt="Drone1" src="https://github.com/IbrahimARahman/AeroVista/assets/108421238/92c2fa99-3824-40dc-a8b9-4eb5d94c6ee6">
</div>



This mirror setup allowed us to manipulate the angle of capture, permitting photos at angles anywhere between straight-on and straight down. Doing this tested the model's ability to detect humans from many different angles, beyond the scope of the SARD_YOLO dataset which only features high angle overhead images.

https://github.com/IbrahimARahman/AeroVista/assets/108421238/9729aa52-dd37-4e73-b004-24f4d16eb650

In order to operate the drone, we used the DJITelloPy API and OpenCV. The DJITelloPy API includes a variety of pre-programmed commands, allowing control over the drone's movement and access to its camera feed. Using this, we created a pre-programmed flight path for the drone, flying overhead and filming us below. The OpenCV library allows for real-time preview of the camera feed, as well as the ability to interface with the drone through keyboard controls. This allowed us to save frame captures on commmand, which we used to populate our validation set.

## Architectures

### Mask R-CNN

<img width="333" alt="MASKRCNNArchSimplified" src="https://github.com/IbrahimARahman/AeroVista/assets/108421238/84a45295-7acb-4db1-b47b-259b3f36e121">
<img width="440" alt="MaskRCNNAarch" src="https://github.com/IbrahimARahman/AeroVista/assets/108421238/fd17d8ba-52b1-4706-8bd1-1730bb47c505">

Mask-RCNN is built off of another CNN called Faster-RCNN. When an image is processed with Faster-RCNN the output consists of class labels and bounding boxes. Since Mask-RCNN is built off of Faster-RCNN, it also outputs both of these but additionally outputs masks, which are pixel-by-pixel mappings of the objects found in the image. This means that most of the architecture is the same, with the only difference being an additional segmentation head in Mask R-CNN, whcih gives it the ability to produce masks.

#### Backbone

The backbone of Mask R-CNN is used to extract feature maps from input images. It consists ResNet-50 and an FPN. ResNet-50 is a 50-layer convolutional neural network which contains 48 convolutional layers, one MaxPool layer, and one average pool layer. Added on top of the ResNet is the Feature Pyramid Network (FPN), which constructs a pyramid of feature maps at multiple levels, enhancing the network’s ability to detect objects at different scales.

#### Inner Layers

Sitting on top of the backbone, the RPN is used to generate object proposals. This layer uses a sliding window over the feature map and outputs a set of potential bounding boxes and their objectness scores indicating the likelihood of an object being present. Anchors of different scales and aspect ratios are used at each sliding position to account for various object sizes and shapes. Once object proposals are made by the RPN, ROI Align is used to extract a fixed-size feature map from each proposal. ROIAlign improves upon the older ROI Pooling used in Faster R-CNN by using bilinear interpolation to compute the exact values of the input features at four regularly sampled locations in each ROI bin, and then aggregating the results using max or average pooling. This method preserves spatial precision, which is crucial for accurate mask prediction.

#### The Head

Mask R-CNN has three parallel heads, the classification head, bounding box regression head, and the mask prediction head. The classicification head determines the probability of each RoI belonging to a specific class. The coordinates of each proposed bounding box are refined by the bounding box regression head. The last head is the mask prediction head. This head is unique from the other two because it is actually a small Fully Convolutional Network that outputs a binary mask for each RoI. The masks are generated independently for each class, but during training and inference, only the mask corresponding to the predicted class (excluding the background) is considered.

### RTMDet

![RTM Det Architecture](https://github.com/IbrahimARahman/AeroVista/assets/108421238/37e4d167-1748-4f32-a7ba-3002bf386585)

The Real-Time Multiple object Detection (RTMDet) archetecture excels in detecting multiple objects with speed and accuracy and in reliable for many computer vision tasks

#### Backbone

The backbone consists of the Cross Stage Partial Network (CSPNet) that reduces computational complexity and enhances inference speed during model training. The CSPNet seperates the feature map that originates from the base layer into 2 segments: a part that goes through a dense block and transition layer while the other combines with the feature map again and is used in the next stage. Unlike a DenseNet Architecture, the gradients that stem from the dense layers are independantly integrated, preventing an excessive amount of duplicate gradient information that come through the map and the layers, removing any type of computational "bottlenecks"

#### Neck

RTMDet uses the Feature Pyramid Attention Module with Positional Encoding for Object Detection (PAFPN) for its neck architecture, fusing multi-level features from the backbone. PAFPN is built on Feature Pyramid Networks (FPNs), popular neural network architectures used for object detection. FPNs come with the disadvantage of having long information paths between lower layers and top-most features. PAFPN overcomes this disadvantage by combining FPNs with bottom-up path augmentation. The neck architecture uses the same basic building blocks as the backbone but includes bottom-up and top-down feature propagation to enhance the overarching pyramid feature map.

#### Head
<img width="400" alt="RTMDetHeadCustom" src="https://github.com/IbrahimARahman/AeroVista/assets/108421238/9b9b6f01-aa6d-4592-8065-16bdb1d1e36a">

The detection head is the last part of the archtecture. It extracts the results from the feature pyramid module (PAFPN) to finally predict the bounding box coordinates and probablities of each potential object in an image. After PAFPN, the heads have the shared convolution weights where combined with a seperated batch normalization layer can predict results for rotated bounding box detection. For instance segmentation like the figure shown above, dynamic kernels(filters) that are generated from the learned weights and parameters can be used to conduct convolution with mask feature maps.

## Results

<img width="400" alt="APARFormula" src="https://github.com/IbrahimARahman/AeroVista/assets/108421238/633d9bfb-8ff9-41b9-a3bc-6dbd18d73f9e">
<img width="374" alt="mAPFormula" src="https://github.com/IbrahimARahman/AeroVista/assets/108421238/c6867db6-bb74-4722-986c-51f62ad5ac95">


So talking about the results, as shown in the figures with the left graph being the Mask-RCNN mAP which was recorded at every 2 epochs and the second one being the RTMDet mAP Curve was recorded at every 1 epoch, we can see that the mAP values increase as more epochs were recorded. When analyzing our table, our mAP values, or the Mean Average Precision(TP/TP+FP), which just to clarify what True Positive is is that when the model detects a human, there is a human. Both box mAP and mask mAP are greater for  RTM-Det than for mask r-cnn which shows us that RTM-Det is more accurate when detecting people, so when RTM-Det says an object is a human, it is usually right about 78% of the time. The reason for choosing mAP is that it is a standardized metric with easy interpretation to determine the model's performance. Furthermore, the average recall(the average ratio of TP to total ground truth positives) essentially tells us how accurate our model is at identifying true positives from all the true positives, again the metric for RTM-Det is greater than Mask r-CNN, although in this case, we would like to see the values for both models be a bit higher. Lastly, when observing the IoU, or the intersection over union, we see that it is quite similar for both models. Our range for the IoU is from 0.5-0.95 and the similarities between the models can be attributed to potentially their similarities in their architecture such as their backbone network or their ROI pooling. 

Below a sample output of RTMDet-Ins-s can be seen.

<img width="1710" alt="RTMDetOutput" src="https://github.com/IbrahimARahman/AeroVista/assets/108421238/eb9633c7-4f59-41a0-be19-f9c60e8f5077">

## Analysis

### Appropriate Fit

<img width="344" alt="MASKRCNNLOSS" src="https://github.com/IbrahimARahman/AeroVista/assets/108421238/e3c6a1b4-8588-4360-b2ef-6e786ec384d9">
<img width="368" alt="RTMDETLOSS" src="https://github.com/IbrahimARahman/AeroVista/assets/108421238/8533480b-13f7-4322-8225-f9cc3c518e3e">

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

In summary, our project aimed to enhance existing machine learning architectures to assist individuals in distress. Moving forward, our goal is to integrate additional approximation or proxy labels into our training procedures to streamline the process. Furthermore, we aspire to expand annotations to encompass various objects within images, like trees or rocks. Additionally, we plan to experiment with image manipulation techniques to simulate different weather conditions and environments by applying diverse filters.

## References

1. Wu, Minghu et al. "Object detection based on RGC mask R-CNN." IET Image Processing, 13 May 2020, 14: 1502-1508. [https://doi.org/10.1049/iet-ipr.2019.0057](https://doi.org/10.1049/iet-ipr.2019.0057).
2. Lyu, Chengqi et al. "RTMDet: An Empirical Study of Designing Real-Time Object Detectors." arXiv, 16 Dec. 2022. [https://arxiv.org/abs/2212.07784](https://arxiv.org/abs/2212.07784).
3. Shastry, Animesh. "SARD_YOLO Dataset." Roboflow Universe, Feb. 2022. [https://universe.roboflow.com/animesh-shastry/sard_yolo](https://universe.roboflow.com/animesh-shastry/sard_yolo).
4. He, Kaiming et al. "Mask R-CNN." arXiv, 24 Jan. 2018. [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870).
5. "SARSAT U.S. Rescues Map." Department of Commerce: SARSAT, 26 Apr. 2024. [https://www.sarsat.noaa.gov/sarsat-us-rescues/](https://www.sarsat.noaa.gov/sarsat-us-rescues/).
