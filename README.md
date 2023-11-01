# CASCADENET: Automatic Segmentation of Vestibular Schwannoma from T1 and T2-Weighted MRI Using Deep Learning

## Table of Contents
1. Introduction
2. Installation
3. Usage
4. Results
5. Dataset
6. Model Architecture
7. Citation
8. License

---

## 1. Introduction

This repository contains the code and documentation for the CascadeNet project, a deep learning model designed for the automatic segmentation of Vestibular Schwannoma (VS) from T1 and T2-weighted Magnetic Resonance Images (MRI). Vestibular Schwannoma is a benign tumor affecting the vestibulocochlear nerve, and its accurate segmentation is crucial for medical diagnosis and treatment planning.

CascadeNet is a two-stage Convolutional Neural Network (CNN) model that provides precise tumor delineation in MRI slices, eliminating the need for manual segmentation, which can be time-consuming and labor-intensive.

The key features of CascadeNet include:
- The first CNN stage generates an initial tumor region estimate.
- The second CNN stage refines the initial estimate using the predicted segmentation mask and input image.
- Spatial attention is incorporated by leveraging the encoder features of the first CNN to enhance segmentation accuracy in the second CNN.

## 2. Installation

To use CascadeNet, you will need the following software and libraries:

- Python (>=3.6)
- PyTorch (>=1.0)
- NumPy
- SimpleITK
- [Your other required libraries and dependencies]

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

## 3. Usage

To run the CascadeNet model for Vestibular Schwannoma segmentation, you can follow these steps:

1. [Step 1: Data Preprocessing]
   - Prepare your MRI data in the appropriate format.
   
2. [Step 2: Model Training]
   - Train the CascadeNet model on your dataset.
   
3. [Step 3: Model Evaluation]
   - Evaluate the model using publicly available or in-house datasets.

For more detailed instructions, please refer to the documentation or code provided in this repository.

## 4. Results

CascadeNet's performance has been evaluated on different datasets and MRI variations. Here are some key results:

- 3D CNN achieved a Dice score of 0.85 for T1-weighted MRI using a public dataset.
- Our proposed method using T1-weighted MRI on the public dataset achieved a Dice score of 0.89.
- In our in-house dataset, the Dice score using a 2D CNN was 0.79.
- Our method achieved a Dice score of 0.83 on the same in-house dataset.

## 5. Dataset

We used publicly available and in-house datasets for model evaluation. Please refer to the "Dataset" section of the code or documentation for more information on these datasets and their sources.

## 6. Model Architecture

For a detailed understanding of CascadeNet's architecture, please refer to the "Model Architecture" section of the code or documentation.

## 7. Citation

If you use CascadeNet in your research or find it useful, please cite the following paper:

```
[Insert Citation Here]
```

## 8. License

This project is licensed under the [License Name], and the details of the license can be found in the "LICENSE" file.

For any questions or issues, please feel free to contact [Your Contact Information].

---

Thank you for using CascadeNet for Vestibular Schwannoma segmentation. We hope this tool will assist in improving clinical workflow and patient management in the field of medical imaging.