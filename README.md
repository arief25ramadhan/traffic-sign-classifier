# **Traffic Sign Recognition** 

## 1. Project Definition

### 1.1. Aim

This project aims to build a classifier that can recognize different traffic signs.

## 1.2. Project Pipeline

The pipeline of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## 2. Data Summary and Exploration

### 2.1. Data Summary

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### 2.2. Data Visualization and Exploration

Here is an exploratory visualization of the data set. 

![alt text][https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/signs.png]

It is a bar chart showing how the data looks like. 

![alt text][https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/class_occurence.png]

## 3. Design and Test a Model Architecture

### 3.1. Image Processing

As a first step, I decided to convert the images to grayscale to minimize memory use for our Neural Network. I think that colors are not crucial for recognizing traffic signs. After that, I normalized the image data to speed up the Neural Network computation.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


#### 3.2. Final Model

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|

I trained the model using a batch size of 128, for 30 epochs, and a learning rate of 0.001.

For my training optimizers I used softmax_cross_entropy_with_logits to get a tensor representing the mean loss value to which I applied tf.reduce_mean to compute the mean of elements across dimensions of the result. Finally I applied minimize to the AdamOptimizer of the previous result.

My final model Validation Accuracy was 0.958

### 3.3. Discussing the Model Selection

My final model results were:
* train set accuracy of 91.9%
* validation set accuracy of 94.6 % 
* test set accuracy of 91.%

I choose the Le Net architecture for our traffic sign classifier back bone. Le Net was developed by Yann Le Cunn. It is the idea  

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

## 4. Test a Model on New Images

### 4.1. Use images found on the web

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

### 4.2. Discuss the model's predictions on these new traffic signs a

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 4.3. Softmax Probability

Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

