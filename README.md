# ASL Alphabet Classification using MLP

This project implements an image classification for American Sign Language (ASL) alphabets using a Multi-Layer Perceptron (MLP) in PyTorch. The project downloads the ASL alphabet dataset from KaggleHub, preprocesses the images, and trains a neural network model to classify the images into 29 classes (A–Z, "del", "nothing", and "space"). Evaluation metrics, confusion matrices, and performance plots are generated to assess the model.

## Features

- **Data Preprocessing:**  
  - Downloads the ASL alphabet dataset using KaggleHub.
  - Reads images from directory structure and resizes them to 64x64 pixels.
  - Assigns labels to images based on folder names.
  
- **Data Splitting:**  
  - Dataset is split into training (80%) and test (20%) sets using scikit-learn's `train_test_split`.
  - Converts the data into PyTorch tensors and creates DataLoaders for batching.

- **Model Architecture:**  
  - Implements an MLP with one input layer, one hidden layer, and an output layer.
  - Uses Batch Normalization, ReLU activations, and Dropout for regularization.

- **Training & Evaluation:**  
  - Trains the model using SGD optimizer and CrossEntropyLoss.
  - Evaluates the model using F1 score, loss metrics, and classification reports.
  - Displays confusion matrices for both training and test sets.
  - Plots training/test loss and F1 scores over epochs.
  - Visualizes a test batch with predicted and ground truth labels.

## Requirements

- Python 3.x
- [kagglehub]([https://github.com/](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)) 
- NumPy
- Pandas
- OpenCV (`cv2`)
- scikit-image
- TensorFlow & Keras (used for dataset utilities)
- PyTorch
- Torchvision
- scikit-learn
- Matplotlib
- Seaborn
