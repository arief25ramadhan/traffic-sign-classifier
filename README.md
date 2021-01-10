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

After loading the data, I visualize some examples from the set to get a sense of the data. Figure 1 displays the images contained in our dataset.

<p align="center">
 <img src="https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/signs.png" width="500">
 <br>
 <em>Figure 1 - Images from the training set</em>
</p>

I then look at the data's class distribution. I draw a bar chart showing the frequency of each class as shown by Figure 2.

<p align="center">
 <img src="https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/class_occurence.png" width="500">
 <br>
 <em>Figure 2 - Training set class distribution</em>
</p>


## 3. Design and Test a Model Architecture

### 3.1. Image Processing

Next, I converted the images to grayscale to minimize memory use for the Neural Network. I think that colors are not crucial for recognizing traffic signs. 

After that, I normalized the image data to speed up the Neural Network computation. Figure 3 illustrates an example of a traffic sign image before and after grayscaling.

<p align="center">
 <img src="https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/rgb_gray.PNG" width="500">
 <br>
 <em>Figure 3 - RGB vs. Grayscale</em>
</p>

### 3.2. Final Model

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5, 6 Filters     	| 1x1 stride, valid_padding, outputs 28x28x6, ReLU activation  |
| Max pooling	      	| 2x2 stride,  valid padding, outputs 14x14x6 				|
| Convolution 5x5, 16 filters	    | 1x1 stride, valid_padding, outputs 10x10x16, ReLU activation |
| Max pooling	      	| 2x2 stride,  valid padding, outputs 5x5x16 				|
| Flatten      	| output 400 (5x5x16)				|
| Fullly connected     	| outputs 120, activation ReLU				|
| Fullly connected     	| outputs 84, activation ReLU				|
| Fullly connected     	| outputs 43, activation SoftMax				|

I trained the model using a batch size of 128, 100 epochs, using Adam optimizer and a learning rate of 0.001. 

### 3.3. Discussing the Model Selection

My final model results were:
* train set accuracy of 91.9%
* validation set accuracy of 94.3% 
* test set accuracy of 91.3%

I choose the Le Net architecture for our traffic sign classifier backbone. Yann Le Cunn developed le Net. Figure 4 displays the architecture of the Le Net network.

<p align="center">
 <img src="https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/LeNet_Original_Image.jpg" width="500">
 <br>
 <em>Figure 4 - Le Net Architecture</em>
</p>
 
 Le Net is straightforward and small, making it perfect to understand the basics of CNNs. Le Net is mostly used as a first step for teaching CNN.

## 4. Test a Model on New Images

### 4.1. Use images found on the web

Figure 5 displays the six German traffic sign images that I found on the web:

<p align="center">
 <img src="https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/six_images.PNG" width="500">
 <br>
 <em>Figure 5 - Six traffic signs images found on the internet</em>
</p>

The first and third image might be difficult to classify because there are similar possibilites such as the 50 and 70 speed limit. Yield is unique but only a few class. The other are chosen at random. 

### 4.2. Discuss the model's predictions on these new traffic signs a

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| Speed limit (60km/h)   			| Speed limit (60km/h) 									|
| Stop					| Stop									|
| Speed limit (30km/h)	      		| Roundabout mandatory				 				|
| Go straight or right			| Go straight or right    							|
| Yield     		| Yield				 				|


The model correctly guessed five of the six traffic signs, which gives an accuracy of 83%, which compares favorably to the test set accuracy of 91%.

### 4.3. Softmax Probability

The code for making predictions on my final model is located in section 3.4. of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five softmax probabilities were. 

<p align="center">
 <img src="https://github.com/arief25ramadhan/traffic-sign-classifier/blob/main/report_images/softmax_prob.PNG" width="500">
 <br>
 <em>Figure 6 - Softmax probability</em>
</p>


The wrong prediction was due to the small number of examples for this kind of image on the data sample. Adding variations of the images by inverting, rotating, or augmenting them might have increased the accuracy.




