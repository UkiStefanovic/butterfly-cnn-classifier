# Butterfly CNN Classifier

This project was developed as part of a Neural Networks course assignment involving Deep Learning. In the project, a Convolutional Neural Network (CNN) classifier is implemented for a dataset of butterfly images.


## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The Butterfly CNN Classifier project aims to classify images of butterflies into 100 different species using deep learning techniques, specifically employing a Convolutional Neural Network (CNN). The model is trained on a dataset sourced from Kaggle, containing images of butterflies across various species.

## Dataset

The dataset used for training and testing the model can be downloaded from [Kaggle - Butterfly Images (100 species)](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species).

## Model

The CNN model architecture used in this project consists of several convolutional layers followed by max-pooling layers to extract features from butterfly images. The final layers are densely connected to classify the images into 100 different butterfly species. The model achieved an accuracy of 81.6% on the validation set.


## Installation

1. Clone the repository:
    git clone https://github.com/UkiStefanovic/butterfly-cnn-classifier.git
    
    cd butterfly-cnn-classifier

2. Install dependencies:
    pip install -r requirements.txt

## Usage
Download the dataset from Kaggle - Butterfly Images (100 species).
Place the dataset in the data/ directory.
Open and run the Jupyter notebook Deep_learning_Project.ipynb to preprocess the data, train the CNN model, and evaluate its performance.
Results
The trained CNN model achieved an accuracy of 81.6% on the validation set. Detailed results, including performance metrics and sample predictions, can be found in the project's documentation.











