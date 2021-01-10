# Traffic Sign Recognition

## 1. Project Definition

### 1.1. Aim

This project aims to build a classifier that can recognize different traffic signs.

## 1.2. Project Pipeline

The pipeline of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train, and test the model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## 2. Data Summary and Exploration

### 2.1. Data Summary

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of the training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### 2.2. Data Visualization and Exploration

After loading the data, we visualize some examples from the set to get a sense of our data. Figure 1 displays the images contained in our dataset.

<p align="center">
 <img src="https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/signs.png" width="500">
 <br>
 <em>Figure 1 - Images from the training set</em>
</p>


We then look at the data's class distribution. Figure 2 is a bar chart showing the frequency of each class. 

<p align="center">
 <img src="https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/class_occurence.png" width="500">
 <br>
 <em>Figure 2 - Training set class distribution</em>
</p>


## 3. Design and Test a Model Architecture

### 3.1. Image Processing

Next, we decided to convert the images to grayscale to minimize memory use for our Neural Network. We think that colors are not crucial for recognizing traffic signs. 

After that, we normalized the image data to speed up the Neural Network computation. Figure 3 illustrates an example of a traffic sign image before and after grayscaling.

<p align="center">
 <img src="https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/class_occurence.png" width="500">
 <br>
 <em>Figure 3 - RGB vs. Grayscale</em>
</p>

### 3.2. Final Model

Our final model consisted of the following layers:

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

We trained the model using a batch size of 128, 30 epochs, and a learning rate of 0.001.

For my training optimizers, we used softmax_cross_entropy_with_logits to get a tensor representing the mean loss value, to which we applied tf.reduce_mean to compute the mean of elements across dimensions of the result. Finally, I applied minimize to the AdamOptimizer of the previous result.

My final model Validation Accuracy was 0.958

### 3.3. Discussing the Model Selection

My final model results were:
* train set accuracy of 91.9%
* validation set accuracy of 94.6 % 
* test set accuracy of 91.%

I choose the Le Net architecture for our traffic sign classifier backbone. Yann Le Cunn developed le Net. It is the idea.

<p align="center">
 <img src="https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/class_occurence.png" width="500">
 <br>
 <em>Figure 4 - Le Net Architecture</em>
</p>
 

## 4. Test a Model on New Images

### 4.1. Use images found on the web

Here are five German traffic signs that I found on the web:

<p align="center">
 <img src="https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/class_occurence.png" width="500">
 <br>
 <em>Figure 5 - Six traffic signs images found on the internet</em>
</p>

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


The model correctly guessed five of the six traffic signs, which gives an accuracy of 83%, which compares favorably to the test set accuracy of 91%.

### 4.3. Softmax Probability

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

The wrong prediction was due to the small number of examples for this kind of image on the data sample. Adding variations of the images by inverting, rotating, or augmenting them might have increased the accuracy.




