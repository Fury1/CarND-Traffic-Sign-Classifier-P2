# **Traffic Sign Recognition**

---

**Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/random-visualization.png "Random Visualization"
[image2]: ./New-German-Signs/ahead_only.jpg "Ahead Only"
[image3]: ./New-German-Signs/pedestrians.jpg "Pedestrians"
[image4]: ./New-German-Signs/speed_limit_20.jpg "Speed Limit 20"
[image5]: ./New-German-Signs/speed_limit_30.jpg "Speed Limit 30"
[image6]: ./New-German-Signs/wild_animal_crossing.jpg "Wild Animal Crossing"
[image7]: ./examples/ahead_only.png "Ahead Only"
[image8]: ./examples/pedestrians.png "Pedestrians"
[image9]: ./examples/speed_limit_20.png "Speed Limit 20"
[image10]: ./examples/speed_limit_30.png "Speed Limit 30"
[image11]: ./examples/wild_animal_crossing.png "Wild Animal Crossing"
[image12]: ./examples/probabilities.png "Probabilities"
[image13]: ./examples/img1lay1.png "Image 1 Layer 1"
[image14]: ./examples/img2lay1.png "Image 2 Layer 1"
[image15]: ./examples/img3lay1.png "Image 3 Layer 1"
[image16]: ./examples/img4lay1.png "Image 4 Layer 1"
[image17]: ./examples/img5lay1.png "Image 5 Layer 1"

---

## Rubric Points
##### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

### Required Files
  * This repository contains all of the required files as specified in the [project rubric](https://review.udacity.com/#!/rubrics/481/view), however, the majority of the content can be viewed by simply loading the jupyter notebook.

---

## Dataset Exploration and Basic Summary
* [x] Dataset Summary
* [x] Exploratory Visualization


**Dataset Summary:**  
Code cell 2 of the jupyter notebook contains a basic summary of the German traffic sign dataset that utilized some basic numpy array functions.

|Type | # |
|:-----:|:---:|
|Training Examples|34799|
|Testing Examples|12630|
|Image Shape|(32, 32, 3)|
|Classes|43|

**Exploratory Visualization:**  
Code cell 3 provides a random visualization of training data as well as the correct corresponding label.

![Random Visualization][image1]

---

## Design and Test a Model Architecture
* [x] Preprocessing
* [x] Model Architecture
* [x] Model Training
* [x] Solution Approach

**Preprocessing:**  
Code cell 4 starts and finishes my data preprocessing. As my only data preprocessing step, I chose to normalize all of my pixel values to a mean of 0 with a low/high of -1/1. I had contemplated converting all of the color images to gray scale then normalizing pixel values to mean of 0, however, I felt that the RGB values in the images could be valuable information for the neural network to pick up on to identify the traffic signs. My thoughts were that the colors would be useful in differentiating signs and I did not want to loose that characteristic.  
Normalizing the RGB pixel values was useful to keep the math simple, prevent the algorithm from searching to much for its loss, and to avoid floating point errors in my overall loss function. The semantic representation of the normalized color image proved to be effective on my training, validation, and test sets.  

The data was read into the jupyter notebook as training, validation, and test sets separately (the data was provided in this format) in code cell 1. From there pixel values for all data sets were normalized with (pixel_value - 128) / 128 for each RGB layer using numpy array slicing and a simple loop in code cell 4.

Initial mean values were calculated before and after the processing.

|Type| Value|
|:-----------------:|:---------------:|
|Starting Pixel Mean|82.794158924612|
|Nomalized Pixel Mean|-0.35317083199818927|  


**Model Architecture:**  
The code for my final model is located in code cell 5.
My final model consisted of an altered version of the LeNet architecture.
The main difference between the standard LeNet neural net and mine is that I have expanded the fully connected layers to double their size, added in max pooling, while also implementing two dropout layers in an attempt to make the final trained model more robust.  

Final Architecture:

| Layer         		|     Description	        					|
|:-----------------:|:---------------------------------:|
| Input         		|32x32x3 RGB image   							|
| Convolution 5x5   |1x1 stride, VALID padding, outputs 28x28x6|
| RELU 1					  |Activation|
| Max Pooling	1(2x2)|2x2 stride,  outputs 14x14x16|
| Convolution 5x5	  |1x1 stride, VALID padding, outputs 10x10x16|
| RELU 2		        |Activation|
| Max Pooling 2(2x2)|2x2 stride,  outputs 5x5x16|
|Flatten Layer      |Prep for first fully connected layer (400,240)|
|RELU 3						  |Activation|
|Dropout            |Dropout layer **(Training Only)**|
|Fully Connected Layer 1|Input 240, output 100|
|RELU 4              |Activation|
|Dropout            |Dropout layer **(Training Only)**|
|Fully Connected Layer 2|Input 100, output 43 **(Final Labels)**|


**Model Training:**  
The code for training the model is located in code cell 6 of the ipython notebook.
To train the final model, I used the following hyperparameters:

|Hyperparameter|Value|
|:---:|:---:|
|Epochs|50|
|Batch Size|256|
|Learning Rate|.001|
|Weight Initial Distribution| Random, mean of 0, sigma of .10|
|Optimizer|Adam Optimizer|

To start off the training all weights were initialized randomly to a mean of 0 with a deviation of .10, while the biases were all set to 0 in the models architecture. The random distribution of the weights with a mean of 0 and a deviation of .10 was chosen because it allowed the network to have the minimal amount of initial gradients on the first pass making the network not very sure of any predictions yet. From there backpropagation could start the learning process and adjust the weights accordingly to minimize the network's loss. If the weights were all initialized to 0 the learning process would struggle, conversely, if the weights were all initialized evenly or with a random distribution that was two large the network my start off being biased to certain features or samples.

When training this model I found that if I ran anywhere from at least 25 to 50 Epochs at a batch size of 64 - 256 and a learning rate of .001 I could achieve sufficient accuracy for the project requirements. For the finally trained model I ran 50 epochs in batches of 256 with a learning rate of .001 and found that the overall trend was improved accuracy throughout the training session. However, after about 30 epochs the gains became minimal and I did not see a reason to train any longer then the 50 epochs in the jupyter notebook for the purposes of this project.

During the lessons in this course we learned and utilized SGD (Stochastic Gradient Decent) as our algorithms optimizer. But I ended up using the more advanced Adam Optimizer that was available in tensorflow. After researching the Adam Optimizer algorithm I came to the conclusion that it would be better to try and use it. The Adam Optimizer automatically integrates momentum by adjusting the learning rate to help with convergence and minimize the loss. One common problem in deep learning training is getting learning stuck at a localized minimum of the error vs the global minimum, momentum was the method mentioned time and time again to combat this. I also thought why not try something new and see how it worked.

**Solution Approach:**  
The code for calculating the accuracy of the model is also located in code cell 6 of the ipython notebook.

|Final Model Results||
|:---:|:---:|
|**Dataset**|**Accuracy**|
|Training|.989|
|Validation|.961|
|Test|.956|

Starting this project I opted for an iterative approach as this was what was used/discussed in lecture and lab. Initially I just trained the LeNet model architecture and with that framework as a starting point my validation accuracy would not get past around .80. For a starting point this was pretty good for a first try but no where near the goal of .93 validation accuracy. More concerning was that my test accuracy much lower then the validation accuracy.

I began by just trying to tweak hyperparameters to see if I could get the network to adapt to this new dataset. However, it seemed to be lacking the robustness needed for real world pictures of traffic signs.
My first thought was that I might be over fitting the network because my training accuracy and testing accuracy had a very large spread from one another.
The first step I took was to add in a drop out layer, I found that this made no difference even after retraining and tweaking hyperparameters again. With that knowledge I thought that the network may be to small for what I was trying to do. In response, I doubled the flattened layer sizes and found a much more improved accuracy with adequate training. After more training and hyperparameter tuning I ended up also adding in two dropout layers amongst the flattened layers. In addition, I added in a calculation for my training and validation accuracy that took into account dropout rates. For my training accuracy I took into account my training dropout rate of .5, however, in my validation accuracy I took into account a dropout rate of 1.0. My thoughts in doing this was that I could compare my accuracies with and without the dropout figure to help me really dial in my hyperparameters. If I could get a solid training accuracy that had half of all my nodes being dropped and my network was working the way I thought it was, my validation accuracy should be higher because I had trained robustness into it. At that point my test set should be in the same arena as my validation and training sets.

After more training and tuning I settled on the mentioned hyperparameters above. I found it really helpful to see training and validation accuracies side by side. This was a pivotal step for me personally that got my model to .956 test accuracy. It also showed me how an over sized network with dropout can be a very robust model. When training initially I could see how the dropout was effecting the overall training accuracy, but after enough epochs the training accuracy catches up and surpasses validation accuracy. This put my test accuracy in the same ballpark as the other training/validation sets.

---

## Test a Model on New Images
* [x] Acquiring New Images
* [x] Performance on New Images
* [x] Model Certainty - Softmax Probabilities

**Acquiring New Images:**  
Here are five German traffic signs that I found on the web:

![Ahead Only][image2] ![Pedestrians][image3] ![Speed Limit 20][image4]
![Speed Limit 30][image5] ![Wild Animal Crossing][image6]

 After resizing (Code cell 8):  
 ![Ahead Only][image7] ![Pedestrians][image8] ![Speed Limit 20][image9]
 ![Speed Limit 30][image10] ![Wild Animal Crossing][image11]

When I went looking for new traffic signs on the web to classify I tried to pick 3 signs that I thought would be easy for the algorithm and 2 signs that would be hard. The first three signs above were in my opinion very clean and concise as well as being very similar to what was supplied in the original training data. The last two signs were harder in the sense that they were not perfectly centered pictures as well as the last sign actually having two signs in it.  
When running these new pictures through my network I figured I would get 3 out of the 5 signs correct. After running it I found that I was getting only 2 out of the 5 correct, yielding a .40 accuracy on these new images (Code cell 11).

The code for making predictions on my final model is located in code cell 10 of the ipython notebook.

Here are the results of the predictions:

| Image|Prediction|
|:----:|:--------:|
|Ahead Only|Ahead Only|
|Pedestrians|Speed limit (30km/h)|
|Speed limit (20km/h)|Speed limit (20km/h)|
|Speed limit (30km/h)|Ahead only|
|Wild Animals Crossing|Priority road|

After seeing these results, it was my assumption that I might not have trained my network enough or that my images are two dissimilar from the training data.
How data is presented to the network is just as important as the network itself. I think it is possible some of the new images are not close enough to the original dataset and it is affecting the performance of the network on these new images. A good test would be to reload the network and train it for more epochs with the same hyperparameters to see if it improves the performance on these new images. If there is no improvement I would like to see if I could improve the performance by performing more preprocessing on the images to see if I could get them initially closer to that of the training images BEFORE normalization.

The starting pixel mean for the training data was 82.794158924612, whereas the new downloaded images pixel mean is 127.76731770833335 (Code cell 9).

That in effect changed the normalized values from -0.35317083199818927 on the training data to -0.0018178313970565797 on the new downloaded images.

Its my thought that these differences could have affected the weighted values of the trained network causing the predictions to be erratic.

**Performance on New Images:**

All of the softmax probabilities for the 43 labels have been outputted in code cell 10, and the top 5 probabilities for each image have been outputted in code cell 12 as shown below.

![Probabilities][image12]

Below is the above output converted into decimal notation vs scientific. Only the top probability is shown below in the chart.

|Image #|Probability|Prediction|Certainty|Certainty Compared to Other 4 Predictions|
|:-----:|:---------:|:--------:|:-------:|:---------------------------------------:|
|1|1.0|Ahead Only|Very Sure|Very Good|
|2|0.92489028|Speed limit (30km/h)|Very Sure|Good|
|3|0.933362544|Speed limit (20km/h)|Very Sure|Good|
|4|0.913291514|Ahead Only|Very Sure|Not Good|
|5|0.41010168|Priority road|Not Very Sure|Not Good|

Overall the predictions revealed from the softmax function show that the first three images had a very good amount of certainty of the prediction made. Unfortunately in the case of Image 2 it was the wrong prediction. Again, this could be possible do to the differences in the images I found vs the data that was provided from the training set, specifically, the averages of pixel values as previously noted.

In the case of images 4 and 5, it appears that the model is not very sure of anything as all of the probabilities are close to each other. There is less of a delta between values showing that these images were hard to classify for the network in its given trained state.

More powerful convolutional layers might have been better able to pick up the off centered signs in the images or more similar images could be trained into the network to combat this.

---

## Visualized Layers

For the final segment of this project as an extra I decided to visualize the layers of my network for each downloaded image. These images prove to be very interesting in the sense of how the network is trying to refine the image to identify features of use to it in each sample as the layers progress through the network.

Ultimately, after many layers a very abstracted view of the image is constructed. After looking at these layers my initial thoughts were that if larger image data was fed into the network instead of (32,32) pixel images, the larger images might be able to yield more dimensions of use to the network. This would also allow the convolutions to get bettered detailed characteristics about the images. However, I'm sure that larger images would also create a harder more costly algorithm to compute.

Images 1 -5 Visualized Through Layer 1 (See code cell 14 in the notebook for the full output):
![Image 1 Layer 1][image13]
![Image 2 Layer 1][image14]
![Image 3 Layer 1][image15]
![Image 4 Layer 1][image16]
![Image 5 Layer 1][image17]

---

## Conclusion and Final Thoughts

Overall I thought this was a very demanding but interesting first project for a deep neural network. There seems to be endless possibilities to build and tune DNNs for a variety of complex hard to solve problems. It has definitely opened my eyes to just some of the possibilities, I look forward to learning more in the near future.
