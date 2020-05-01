# Dog Breed Classifier in PyTorch
This is a repo for the Dog Breed Classifier Project  in Udacity Nanodegree

It is implemented by using PyTorch library.

**Udacity's original repo is [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification)**



## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI  Nanodegree! In this project, you will learn how to build a pipeline that  can be used within a web or mobile app to process real-world,  user-supplied images.  Given an image of a dog, your algorithm will  identify an estimate of the canine’s breed.  If supplied an image of a  human, the code will identify the resembling dog breed.

[![Sample Output](https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_dog_output.png)](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/project-dog-classification/images/sample_dog_output.png)

Along with exploring state-of-the-art CNN models for classification  and localization, you will make important design decisions about the  user experience for your app.  Our goal is that by completing this lab,  you understand the challenges involved in piecing together a series of  models designed to perform various tasks in a data processing pipeline.   Each model has its strengths and weaknesses, and engineering a  real-world application often involves solving many problems without a  perfect answer.  Your imperfect solution will nonetheless create a fun  user experience!



## Import Datasets

* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* Download the [human_dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

## Dependencies
* pytorch
* numpy
* matplotlib
* cv2
* pillow

## CNN Structures (Building a model on my own)

Net(


  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
  
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  
  (fc1): Linear(in_features=50176, out_features=500, bias=True)
  
  
  (fc2): Linear(in_features=500, out_features=133, bias=True)
  
  
  (dropout): Dropout(p=0.25)
  
  
  (batch_norm): BatchNorm1d(500, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  
  
)


### Medium story related to the project can be found [here](https://medium.com/@anantagarwalcourses/how-to-develop-a-cnn-classifier-from-scratch-3d5aef9e24b2)

## Model performance

Final model metrics are:
* Testing Loss Average: 1.051463 
* Test Accuracy: 80% (670/836)

## Reflection

In the project we starts with opject detection using haarlike features with the implementation of openCV and ended up developing a CNN based model for classification. In between we touched transfer learning ( tunning pre trainded model for our purpose) and also developing a CNN model right from scratch.

The most interestng part in the whole excercise was the need to transform the image to the specific shape for CNN. I always wonder AI is developing soo fast and our deep learning models are achieveing soo much but still all the things can shop working by just changing 1 pixel in model input.
But pytorchs' transform functions always comes to rescue to maintain consistency in transformation.







