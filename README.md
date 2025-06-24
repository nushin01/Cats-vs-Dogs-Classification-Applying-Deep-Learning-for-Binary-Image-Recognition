# Cats-vs-Dogs-Classification-Applying-Deep-Learning-for-Binary-Image-Recognition
The goal is to build a model that distinguishes between images of cats and dogs — a binary classification task.

## Cats vs Dogs Classification using Deep Learning

A deep learning project applying convolutional neural networks (CNNs) and transfer learning to classify cat and dog images from the Kaggle "Dogs vs. Cats" dataset.

## Project Overview

This project tackles a classic computer vision problem: binary image classification. Using TensorFlow and Keras, we aim to train models that accurately distinguish between images of cats and dogs.

We compare a custom-built CNN with a transfer learning approach using MobileNetV2, analyzing model performance through training curves, confusion matrices, and classification reports.

---

## Dataset

- Name: Dogs vs. Cats
- Source: [Kaggle Competition Page](https://www.kaggle.com/competitions/dogs-vs-cats)
- Images: 25,000 JPG images (12,500 cats + 12,500 dogs)
- Format: Labeled via filenames (e.g., `cat.0.jpg`, `dog.1.jpg`)

---

## Deep Learning Models

### 1. Custom CNN (From Scratch)
- 3 convolutional layers
- Flatten → Dense layers
- 10 epochs
- ~4.8M parameters

### ✅ MobileNetV2 (Transfer Learning)
- Pretrained on ImageNet
- Frozen base model + new head
- Global average pooling
- 10 epochs
- ~2.2M parameters

---

## Results Summary

| Metric                | Custom CNN     | MobileNetV2     |
|----------------------|----------------|------------------|
| Validation Accuracy | 81.04%        | 96.30%       |
| Validation Loss     | 0.8322        | 0.1431       |
| Training Time       | ~55 mins      | ~33 mins     |
| Params              | ~4.8M         | ~2.2M**        |
| Generalization      | Overfitted    |  Excellent     |

> **Conclusion:** MobileNetV2 outperformed the custom CNN in terms of accuracy, speed, and generalization.

---

## Visualizations

- Sample images preview
- Image size distribution
- Training vs validation accuracy/loss plots
- Confusion matrices
- Classification reports

---

## Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, pandas, matplotlib, seaborn
- Scikit-learn
- PIL

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cats-vs-dogs-classification.git
   cd cats-vs-dogs-classification

2. Set up your environment:

    pip install -r requirements.txt

3. Download the dataset from Kaggle and extract it under a folder named ./dogs-vs-cats/train.

4. Run the notebook:

   jupyter notebook Cats_vs_Dogs_Classification.ipynb


## References

Kaggle – Dogs vs. Cats Dataset

Microsoft Research Asirra Dataset (Original Source)

MobileNetV2 Paper

TensorFlow Documentation – https://www.tensorflow.org/

Keras API Reference – https://keras.io/api/

## Author
Nushin Anwar

## License
This project is under Kaggle Terms of Use.
