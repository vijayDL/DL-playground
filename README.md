# DL-playground
Deep Learning notes.

## Background
* [Setup - Software, Hardware](https://github.com/vijayDL/DL-playground/blob/master/background/Setup%20-%20Software%2C%20hardware.ipynb): Software and Hardware setup for DL.
* [Introduction to Python, numpy, Pandas, matplotlib](https://github.com/vijayDL/DL-playground/blob/master/background/Python%20tutorials.ipynb): Crash cource on python, and other popular libraries for ML/DL.

## Intro
 * [Introduction to Neural networks](https://github.com/vijayDL/DL-playground/blob/master/intro/1.%20Intro%20to%20Neural%20Networks.ipynb):
    How does NN classify non linear data?. Is it a stacked ensemble of linear classifiers?.
 * [Understanding gradient flows](https://github.com/vijayDL/DL-playground/blob/master/intro/2.%20Gradient%20Flow.ipynb):
    Understanding Backpropagation using gradient flow.
    
## CNN
 * [Understanding convolution](https://github.com/vijayDL/DL-playground/blob/master/cnn/1.%20Understanding%20Convolution.ipynb):  Implementing the convolution from scratch, and comparing it with tensorflow implementation.
 * [Understanding convolution part 2](https://github.com/vijayDL/DL-playground/blob/master/cnn/1a.%20Understanding%20Convolution%20Network%20-%20Part2.ipynb):
   Compare convolutions using a predefined filter like gabor with a learnt filter using backpropagation.
 * [Transfer Learning](https://github.com/vijayDL/DL-playground/blob/master/cnn/2.%20Transfer%20Learning%20inception_resnet_v2.ipynb):
   Learn to use inception Resnet V2 architecture for transfer learning.
 * CNN Architectures:
     * [Understanding Resnet](https://github.com/vijayDL/DL-playground/tree/master/cnn/cnn_architectures/resnet):
      Paper summary of [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf).
     * [Wide resnet on CIFAR 10](https://github.com/vijayDL/DL-playground/tree/master/cnn/cnn_architectures/wide-resnet):
         Classification of CIFAR 10 dataset using wide resnet. Contains input data augmentaion using the Tensorflow Dataset API, 
         slim based wide resnet implementaion, achieving an accuracy of **~93.23%.**
     * [U-net Segmentation of Medical images](https://github.com/vijayDL/DL-playground/tree/master/cnn/cnn_architectures/segmentation): A TF-slim implementaion of U-net arachitecture for segmenting White blood cells on Medical images.       
     <img src='https://github.com/vijayDL/DL-playground/blob/master/cnn/cnn_architectures/segmentation/result.png' height='400' width='600'>   
  
## RNN
 * [Recurrent Neural Network](https://github.com/vijayDL/DL-playground/blob/master/rnn/1.%20Recurrent%20Neural%20Networks.ipynb)
   Character level RNN implementation and experiments on state variables.
 * [Exploding gradients in RNN](https://github.com/vijayDL/DL-playground/blob/master/rnn/2.%20Exploding%20gradient%20.ipynb):
   Understanding the exploding and vanishing gradient problem in RNNs.
   
   
 ## Environment:
 The above code was developed on Linux using *Python 3.5*, *Tensorflow 1.8*.
