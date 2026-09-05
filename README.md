# Bank Document Classification with CNN

A lightweight **Convolutional Neural Network (CNN)** built with **PyTorch** to classify documents into four categories:

- Invoice
- Form
- Memo
- Budget

## Dataset
Used the **RVL-CDIP** validation dataset with a balanced subset:
https://www.kaggle.com/datasets/ananthakrishnanpv/rvl-cdip-validation-dataset

- **8,000 images**
- 2,000 images per class
- Train: 5,600
- Validation: 1,200
- Test: 1,200

## Model

Custom **4-block CNN**:

`Conv2D → ReLU → MaxPool` × 4  
`Flatten → Linear(128) → Dropout(0.5) → Linear(4)`

Input: **256 × 192 grayscale images**

## Training

The final model was trained using:

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Device | NVIDIA Tesla T4 |
| Input size | 256 × 192 |
| Channels | 1 (grayscale) |
| CNN blocks | 4 |
| Filters | 32 → 64 → 128 → 256 |
| Batch size | 32 |
| Optimizer | Adam |
| Initial learning rate | 0.0005 |
| Dropout | 0.5 |
| Weight decay | None |
| Batch Normalization | None |
| Data augmentation | None |
| Scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 2 |
| Early stopping patience | 4 |
| Maximum epochs | 20 |


## Experiments

Compared different architectures, resolutions and regularization techniques.

| Experiment | Best Validation Accuracy |
|---|---:|
| 3-Block CNN | 70.67% |
| 4-Block CNN | 73.50% |
| + Scheduler | 75.50% |
| + Scheduler + Early Stopping | **76.33%** |

The best input resolution was **256 × 192**.


### Additional Experiments

| Experiment | Best Validation Accuracy |
|---|---:|
| Batch Normalization | 55.67% |
| Adaptive Global Average Pooling | 52.42% |
| Data Augmentation | 60.08% |
| Dropout 0.3 | 74.08% |
| Dropout 0.5 | **76.33%** |

These experiments showed that the final architecture benefited from preserving spatial document information without Batch Normalization, aggressive augmentation, or global average pooling.


## Results

**Validation Accuracy:** 76.33%  
**Test Accuracy:** **75.42%**

Best performing class: **Memo — F1 84.39%**

Most challenging class: **Form — F1 68.51%**

The model showed reasonable generalization, with only a small gap between validation and test performance.

## Key Findings

- Increasing CNN depth from 3 to 4 blocks improved performance.
- `256 × 192` performed better than larger tested resolutions.
- `ReduceLROnPlateau` improved convergence.
- Dropout `0.5` performed better than `0.3`.
- Batch Normalization and aggressive augmentation reduced performance in this setup.
- Document layout information was important for classification.


## Tech Stack

**Python · PyTorch · Torchvision · NumPy · Pandas · Scikit-learn · Matplotlib · Pillow**

### Final Result

> **Custom 4-Block CNN — 75.42% Test Accuracy**
>
> **76.33% Best Validation Accuracy**
