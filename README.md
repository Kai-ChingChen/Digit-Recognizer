# Digit-Recognizer

A handwritten digit(0-9) recognizer built with a custom numpy-based automatic diferentiation engine inspired by Andrej Karpathy's youtube video: The spelled-out intro to neural networks and backpropagation: building micrograd.

The project implements the data exploration, neural network model, training, and inference from scratch, aiming to have a thorough understanding of how neural network is built and trained from the most fundamental levels. 

## Model and Training

Architecture: **784 -> 64 -> 64 -> 64 -> 10**, with `tanh` as the activation function of each hidden layer.

Inference: The model takes 28x28 pixels single digit input and predict the highest probablity among the probability distribution across 10 digits.

## Results
Untrained MLP, seed 42 | 11.06%

Trained MLP | 96.65%