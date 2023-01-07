"""
Traffic Signs Classification Project - LeNet Deep Network

This project utilizes the LeNet deep network architecture to classify 42 different 
types of traffic signs. LeNet refers to a convolutional neural network that can be 
used for computer vision and classification models. This project showcases a 
step-by-step implementation of the model as well as in-depth notes to customize the 
model further for higher accuracy.

We will first start by importing the necessary libraries we will need:
"""
# Classic libraries that will help us read and analyze data
import numpy as np
import pandas as pd

# Libraries used for plotting and data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Using pickle package to open our data, not much use after that
import pickle


"""
Importing the data into three different sections: training, validation, and
testing. Training data is used to train the network, testing data is used to test
the trained network with data that it has never seen to see how it would perform in
the real world. The validation dataset ensures that we avoid overfitting by showing
the network validation data every epoch (or run) in a process called 
cross-validation to ensure that the network is not focusing on the details of the
training data. 
"""
with open("./traffic-signs-data/train.p", mode = 'rb') as training_data:
  train = pickle.load(training_data) # Use pickle's load method to load the defined data as the variable test

with open("./traffic-signs-data/valid.p", mode = 'rb') as validation_data:
  valid = pickle.load(validation_data)

with open("./traffic-signs-data/test.p", mode = 'rb') as testing_data:
  test = pickle.load(testing_data)


"""
Splitting the data into our individual training and testing set variables.
"""
# Assigning the features of the training set as X_train and the dependent variable (labels) as y_train
X_train, y_train = train['features'], train['labels']

# Assigning the features of the validation set as X_validation and the dependent variable (labels) as y_validation
X_validation, y_validation = valid['features'], valid['labels']

# Assigning the features of the testing set as X_test and the dependent variable (labels) as y_test
X_test, y_test = test['features'], test['labels']

# Checking the dimensions of the training set
X_train.shape # Gives us an output of a four element tuple. The first number is the quantity of images, the second is the width of image in pixels, the third is the height of the image, and the last number the depth - in this case the 3 tells us that the images are colored since they are being multiplied for both Red, Green, and Blue
y_train.shape # Gives us a tuple of one element, which is a label for each image in the training set

X_validation.shape
y_validation.shape

X_test.shape
y_test.shape


"""
Using matplotlib to display a randomly choosen image and see if it matches with its 
label, just to see what the images are and how the network will be classifying them. 
Each type of sign has its own class which will identify the type of sign to the 
model. 
"""
i = 23 # The index of the image we want to look at, arbitrarily chose 23

plt.imshow(X_train[i]) # Using matplotlib's .imshow method to show an image from the features training set at the index of 23
y_train[i] # Also show us the corresponding label from the same index in the label's training set - the label tells us that the sign is a "End of No Passing"

plt.imshow(X_validation[i]) # Verifying images for the validation dataset
y_validation[i]

plt.imshwo(X_test[i]) # Verifying images for the testing dataset
y_test[i]