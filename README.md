# LULC Semantic Segmentation with U-Net + ResNet34

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Mean IoU](https://img.shields.io/badge/Mean%20IoU-0.7911-green.svg)]()

This project uses U-Net with a ResNet34 backbone to classify Land Use Land Cover (LULC) from satellite imagery, developed during my ISRO internship. It segments 4 classes (Background, Buildings, Woodlands, Water) and achieved a 0.84 Mean IoU on validation with batch evaluation, and 0.79 above on the full test dataset. Inspired by the idea of AI augmenting human geospatial insights, this work reflects a passion for technology that empowers.

![Architecture Diagram](Results/system_architecture.png) 

## Overview
Built on the landcover.ai dataset, this project automates LULC mapping for urban planning and environmental monitoring. It handles 21k high-resolution images, patched to 256x256, with augmentation to tackle class imbalance.

- **Dataset**: [Explore the full dataset here](https://www.kaggle.com/datasets/adrianboguszewski/landcoverai) (70% train, 20% val, 10% test).
- **Training**: `notebooks/training_with_batch_evaluation.ipynb` trains the model and evaluates on batches.
- **Full Evaluation**: `src/final_lulc_evaluation.py` assesses the entire dataset.
- **Data Preparation**: `src/data_prep.py` patches and splits the data.

## Installation
```bash
git clone https://github.com/Faheem-02/LULC-Classification-U-Net-ResNet34.git
cd LULC-Classification-U-Net-ResNet34
pip install -r requirements.txt
