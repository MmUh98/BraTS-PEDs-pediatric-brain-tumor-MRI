# Comprehensive Analysis of Pediatric High-Grade Gliomas: From Classical Radiomics to 3D Vision Transformers

**Authors:** Muhammad Mansoorulhaq

**Dataset:** [BraTS-PEDs (Brain Tumor Segmentation in Pediatric Magnetic Resonance Imaging)](https://doi.org/10.7937/DX5C-TJ86) 

---

## 📌 Project Overview

This project presents an end-to-end computer vision and medical image analysis pipeline for the segmentation and classification of pediatric brain tumors. Pediatric high-grade gliomas, particularly diffuse midline gliomas (DMGs), have a dismal prognosis, with five-year survival rates below 20%.

Due to strict hardware constraints (Google Colab, ~15GB GPU VRAM) and a massive dataset size of 32.7GB, this project implements highly optimized memory-mapped processing. The methodology spans classical edge detection, handcrafted radiomic feature classification, and state-of-the-art 3D Vision Transformers (UNETR).

## 📊 Dataset Details

* 
**Name:** BraTS-PEDs (Pediatric subset) 


* 
**Subjects:** 457 pediatric patients with high-grade gliomas. We utilized the 257-patient training set containing ground truth masks.


* 
**Modality:** 3D Multiparametric MRI (mpMRI) 


* Pre-contrast T1-weighted (T1) 


* Post-contrast T1-weighted (T1CE) 


* T2-weighted (T2) 


* T2-weighted fluid-attenuated inversion recovery (T2-FLAIR) 




* **Annotations (RAPNO Guidelines):**
* 
**Label 1:** Enhancing Tumor (ET) - Areas showing contrast enhancement on T1CE relative to T1.


* 
**Label 2:** Cystic Component (CC) - Fluid-filled tumor regions, hyperintense on T2 and hypointense on T1CE.


* 
**Label 3:** Peritumoral Edema (ED) - Hyperintense signal on FLAIR scans, appearing as finger-like spread.


* 
**Label 4:** Non-Enhancing Tumor (NET) - Abnormal signal intensity within the tumor not classified as ET or cystic.





---

## 🚀 Pipeline Phases

### Phase 1: Volumetric Preprocessing & Memory Management

* **Lazy Loading:** Utilized `monai.data.Dataset` to stream NIfTI blocks from disk to GPU on-demand, preventing Out-Of-Memory (OOM) crashes.
* **Z-Score Normalization:** Channel-wise Z-score normalization computed strictly on foreground brain tissue (`nonzero=True`).
* **3D Spatial Patching:** Extracted randomized 96x96x96 patches with a 1:1 positive/negative sampling ratio to mitigate class imbalance.

### Phase 2: Structural Geometry and Classical Masking

* **Edge Detection:** Explored Canny and Sobel edge detectors, identifying their limitations in semantic medical segmentation.
* **Morphological Processing:** Applied Erosion, Dilation, Opening, and Closing pipelines to clean and smooth expert binary masks.
* **Shape Descriptors:** Extracted 8-direction Freeman chain codes, first difference, and shape numbers for boundary analysis. Computed a Convexity Ratio (0.712) indicating irregular tumor growth patterns.

### Phase 3: Radiomic Feature Extraction & Classical ML

* **Texture Analysis:** Computed Gray-Level Co-occurrence Matrices (GLCM) extracting Contrast, Correlation, Energy, Homogeneity, and Entropy.
* **Classification Task:** Binary classification of slices (Enhancing Tumor present vs. absent).
* **Models Evaluated:** Support Vector Machines (SVM - RBF Kernel), Random Forest, Gradient Boosting.
* **Key Finding:** SVM achieved **80% Accuracy**, with Random Forest feature importance identifying **GLCM Correlation** as the strongest predictor of active tumor tissue.

### Phase 4: 3D Vision Transformer (UNETR) Segmentation

* **Architecture:** 3D UNETR mapping 3-channel 3D inputs (T1CE, T2, FLAIR) via Multi-Head Self-Attention.
* **Optimization:** Overcame VRAM limits using Mixed Precision (`autocast`) and Gradient Accumulation (simulating a batch size of 8 using `BATCH_SIZE = 2` and `ACCUM_STEPS = 4`).
* **Loss Function:** Dice Cross-Entropy (DiceCELoss).
* **Results:** Achieved a validation Mean Dice Score of **0.4608** over 50 epochs trained entirely from scratch.

---

## ⚙️ Environment & Dependencies

The pipeline heavily relies on the **MONAI** (Medical Open Network for AI) framework for data transforms and the UNETR architecture.

```bash
pip install torch torchvision torchaudio
pip install monai nibabel numpy scipy matplotlib scikit-learn scikit-image

```

## ⚠️ Known Dataset Anomalies

During training, a limited number of cases from the BraTS-PEDs dataset were identified with issues  and were explicitly excluded:

* 
`BraTS-PED-00024-000` (Training Set): Skull-stripped images.


* 
`BraTS-PED-00098-000` (Training Set): t1c image is skull-stripped.


* 
`BraTS-PED-00281-000` (Validation Set): Different orientation.


* 
`BraTS-PED-00287-000` (Validation Set): Head Tilt.



---

## 📝 Citation

If utilizing this dataset or methodology, please cite the primary BraTS-PEDs dataset:

> Fathi Kazerooni, A., et al. (2025). The Brain Tumor Segmentation in Pediatric Magnetic Resonance Imaging (BraTS-PEDs) (Version 1) [Data set]. The Cancer Imaging Archive. [https://doi.org/10.7937/DX5C-TJ86](https://doi.org/10.7937/DX5C-TJ86) 
> 
>
