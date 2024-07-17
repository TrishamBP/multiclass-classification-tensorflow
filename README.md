# Fashion MNIST Classification with TensorFlow

This project demonstrates the process of building and optimizing a neural network model to classify images from the Fashion MNIST dataset using TensorFlow and Keras.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Learning Rate Optimization](#learning-rate-optimization)
7. [Model Evaluation](#model-evaluation)
8. [Results](#results)
9. [Visualizations](#visualizations)
10. [Future Work](#future-work)
11. [Dependencies](#dependencies)

## Introduction

This project aims to classify clothing items from the Fashion MNIST dataset using various neural network architectures and optimization techniques. We explore the impact of data normalization, model complexity, and learning rate scheduling on model performance.

## Dataset

We use the Fashion MNIST dataset, which consists of 70,000 grayscale images of 10 categories of clothing items. The dataset is split into 60,000 training images and 10,000 test images.

Categories:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Model Architecture

We experiment with multiple model architectures:

1. Basic model (Model 1):
   - Flatten layer
   - Dense layer (4 units, ReLU activation)
   - Dense layer (4 units, ReLU activation)
   - Output layer (10 units, softmax activation)

2. Deeper model (Model 2, 3, 4):
   - Flatten layer
   - Dense layer (4 units, ReLU activation)
   - Dense layer (4 units, ReLU activation)
   - Dense layer (4 units, ReLU activation)
   - Output layer (10 units, softmax activation)

## Data Preprocessing

We normalize the pixel values of the images by dividing them by 255, scaling them to the range [0, 1].

## Model Training

We train the models using the following configuration:
- Loss function: Sparse Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy
- Epochs: 10-40 (varying by experiment)

## Learning Rate Optimization

We implement a learning rate scheduler to find the optimal learning rate for our model. The learning rate is increased exponentially over 40 epochs, and we plot the learning rate vs. loss to identify the best learning rate.

## Model Evaluation

We evaluate our models using the following metrics:
- Training and validation accuracy
- Training and validation loss
- Confusion matrix
- Random image prediction visualization

## Results

Here are the accuracy results for our best performing model (Model 4):

| Metric | Value |
|--------|-------|
| Training Accuracy | 81.71% |
| Validation Accuracy | 80.3% |

## Visualizations

### Learning Curves

![image](https://github.com/user-attachments/assets/b3c81f95-46f6-4dc5-8c70-d2a3ae8e4519)

### Learning Rate vs. Loss

![image](https://github.com/user-attachments/assets/6d435c90-c70e-48ea-bd67-139a1a2baec0)

### Confusion Matrix

![image](https://github.com/user-attachments/assets/c999201a-5da1-447c-9816-458e334ecd07)

### Random Image Prediction

![image](https://github.com/user-attachments/assets/3809c577-8ca7-411a-b9a8-50d1997b4e24)

## Future Work

- Experiment with convolutional neural network (CNN) architectures
- Implement data augmentation techniques
- Try transfer learning with pre-trained models
- Explore ensemble methods for improved accuracy

## Dependencies

- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
