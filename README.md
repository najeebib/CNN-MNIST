# MNIST Digit Classification Convolutional Neural Network  

## Overview  
This repository contains a Convolutional Neural Network (CNN) implemented in PyTorch for the task of classifying handwritten digits from the MNIST dataset.    

## Data   
The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0 to 9). It is widely used for benchmarking machine learning models.    

## Model Architecture  
The CNN architecture consists of two convolutional layers with batch normalization and ReLU activation, followed by max-pooling layers. The output is then flattened and passed through two fully connected layers with dropout for regularization.  

## Initialization  
Weights in the convolutional layers are initialized using the Kaiming normal initialization.  

## Training  
The model is trained for 10 epochs with a batch size of 100. The Adam optimizer is used with a learning rate of 0.01, and Cross Entropy Loss is employed as the loss function.  

## GPU Support  
The code checks for the availability of a CUDA-compatible GPU and moves the model to the GPU if present.  

## Evaluation  
After training, the model is evaluated on the test set. The accuracy of the model on the test set is calculated, providing a performance metric for the trained CNN.  

## Results  
The script prints and plots the training and testing loss, as well as the training and testing accuracy over the epochs.  

## Inference   
A random image from the test set is chosen, and the trained model predicts its label. The predicted label is compared with the actual label.  

 
