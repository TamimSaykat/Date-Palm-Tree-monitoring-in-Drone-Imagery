## Drone-Based Monitoring of Date Palm Trees Using a Self-Supervised and Semi-Supervised YOLOv12s Backbone


| | | |
|---|---|---|
| ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=0B1220) | ![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white&labelColor=0B1220) | ![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=flat-square&logo=nvidia&logoColor=white&labelColor=0B1220) |
| ![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-FF6F00?style=flat-square&logo=github&logoColor=white&labelColor=0B1220) | ![Self--Supervised](https://img.shields.io/badge/Learning-Self--Supervised-0EA5E9?style=flat-square&labelColor=0B1220) | ![Semi--Supervised](https://img.shields.io/badge/Learning-Semi--Supervised-8B5CF6?style=flat-square&labelColor=0B1220) |

### ABSTRACT
Date palm trees are a key economic crop in arid regions, and accurately locating and monitoring individual trees and their health status is crucial for yield forecasting and plantation management. In this work, we address palm-tree object detection and health classification in UAV imagery using the Dat Palm Fx dataset, which contains 4,802 annotated images with three health classes (healthy, abnormal, dead). We take YOLOv12s as a fully supervised baseline detector, and further investigate a semi-supervised approach (Soft Teacher) and two self-supervised pretraining strategies (BYOL and SimCLR) for improving detection performance under limited labeled data. Our final model couples BYOL pretraining with a YOLOv12s backbone and
is fine-tuned on the labeled portion of Dat Palm Fx. The proposed BYOL–YOLOv12s detector attains a mean precision of 0.9317, recall of 0.8994, mAP@0.50 of 0.9609, and mAP@0.50:0.95 of 0.7255 on the validation set, and 0.9078 precision,0.9070 recall, 0.9569 mAP@0.50, and 0.6925 mAP@0.50:0.95 on the held-out test set. Compared with the supervised baseline and the semi-supervised and alternative self-supervised variants, the BYOL-based model consistently yields the best trade-off between precision and recall. Finally, we deploy the trained detector in a web-based monitoring system that visualizes detections and healt  maps from drone imagery, providing a practical and scalable tool for automated palm plantation health monitoring.


### Methodology

This repository implements a complete UAV-based pipeline for date palm tree detection and health-status monitoring using a YOLOv12s detector enhanced with self-supervised (BYOL, SimCLR) and semi-supervised (Soft Teacher) learning strategies.The overall research workflow, illustrated in 1, presents the complete pipeline for the proposed approach: 

![Workflow Diagram](assets/workflow.jpg)

### Dataset

We use the **Dat Palm Fx** dataset (Roboflow Universe), consisting of **4,802 RGB UAV images** annotated with axis-aligned bounding boxes. Each palm instance belongs to one of three health-condition classes:

* **healthy_palm**
* **abnormal_palm**
* **dead_palm**

### Class Distribution

<p align="center">
  <img src="https://raw.githubusercontent.com/TamimSaykat/Date-Palm-Tree-monitoring-in-Drone-Imagery/main/assets/class%20distribution.jpg" width="520" alt="Per-class distribution">
</p>


### Data Augmentation

We apply Albumentations-based augmentation **only on the training split** to improve robustness against:

* viewpoint/scale variation
* illumination changes
* cluttered backgrounds (soil, shadows, buildings)


| Split     | Original | Augmented |     Extra |   Labels |
| --------- | -------: | --------: | --------: | -------: |
| Train     |     3855 |      5355 |     +1500 |     5355 |
| Valid     |      200 |       200 |        +0 |      200 |
| Test      |      747 |       747 |        +0 |      747 |
| **Total** | **4802** |  **6302** | **+1500** | **6302** |

### Models & Implementation

🧱 Backbone benchmarking  
We first trained three lightweight detectors: YOLOv10s, YOLOv11s, and YOLOv12s using the same
dataset split, input size (640 × 640 ), and training setup to ensure a fair comparison. Among the tested backbones, YOLOv12s
achieved the best overall detection performance and showed more stable learning behavior. Therefore, we chose YOLOv12s as
the baseline backbone for the rest of this study.

**Figure:** YOLOv12s architecture used as the baseline detector backbone.
![YOLOv12s](assets/yolov12s.png)
🧠 Learning strategies  
After selecting YOLOv12s, we explored methods to further improve generalization under plantation
conditions (e.g., changes in viewpoint, illumination, canopy density, and background). We evaluated (i) a semi-supervised
detection method (Soft Teacher) and (ii) two self-supervised learning (SSL) methods (BYOL and SimCLR). For SSL, we
removed the detection head and used the YOLOv12s backbone as an encoder to learn strong visual representations from
augmented views of the same image. After pretraining, we re-attached the detection head and fine-tuned the full detector on the
labeled training set.

🧩 Proposed model  
In our experiments, BYOL produced the most effective representations and achieved the highest
accuracy after fine-tuning. For this reason, our proposed system uses a BYOL-pretrained YOLOv12s backbone as the final
detection model.

**Figure:** BYOL pretraining architecture used to initialize the YOLOv12s backbone. The online network (encoder–projector–predictor) learns to match the target network (encoder–projector), updated via EMA, across two augmented views of the same UAV image.
![BYOL Self-Supervised Learning Architecture](assets/Figure5.png)




