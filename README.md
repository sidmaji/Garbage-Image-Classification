# Garbage-Image-Classification

### Real-Time Waste Classification using MobileNetV3

This repository contains my contributions to the **Garbage-Crew** project built during the **2025 UTD Deep Dive AI Summer Workshop**. It focuses on the **image classification** component of an AI-powered smart waste sorter.

---

## What This Repo Includes

-   **Dataset Preparation**: Preprocessing and synthesis of a unified waste classification dataset from multiple public sources.
-   **Model Training**: Fine-tuned a pretrained **MobileNetV3** model on 8 waste categories using PyTorch and `timm`.
-   **Real-Time Classification**: Python script using **OpenCV** and the trained model to classify waste items in a live webcam feed.

---

## Dataset

Unified across multiple sources and organized into 8 classes:

-   `battery`, `glass`, `metal`, `organic_waste`, `paper_cardboard`, `plastic`, `textiles`, `trash`

Available on Kaggle:
[Unified Waste Classification Dataset](https://www.kaggle.com/datasets/siddhantmaji/unified-waste-classification-dataset)

---

## Model

-   **Architecture**: `mobilenetv3_large_100` from `timm`
-   **Input Size**: 224x224
-   **Framework**: PyTorch
-   **Loss**: CrossEntropyLoss
-   **Optimizer**: Adam
-   **Training**: 80/20 stratified split with basic preprocessing (resize, normalize)

---

## Live Webcam Classifier

-   Uses your system’s webcam to classify waste in real-time
-   Optional object detection feature using YOLOv8 for bounding boxes
-   Snapshot mode: Press 's' to save the current frame with classification results

---

## Requirements

-   Python 3.9+
-   `torch`, `timm`, `opencv-python`, `torchvision`, `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Full Project

This repository focuses only on **data + model + real-time classification**.

For the **complete system** including hardware integration (EV3 motor, Raspberry Pi, physical trapdoor mechanism), visit:

[Garbage-Crew/Garbage-Crew](https://github.com/Garbage-Crew/Garbage-Crew)
