# VanGoghMLModel

For information on the dataset used to train this model and how it was obtained, please see the [VanGoghScraping repository](https://github.com/antoine-carpentier/VanGoghScraping).



The goal of this model is to create a Convolutional Neural Network and train it to classify paintings by Vincent Van Gogh.   

The dataset used to train our model is comprised of 210 Van Gogh paintings and 437 paintings by others renowned artists, for a total of 681 labeled datapoints.
Our dataset will be split into training and validation sets with a 70/30 split.

The CNNs in this project follow LeNet5's design with the following improvements:
- The single 5x5 filters are replaced by two stacked filters (3x3 and 5x5).
- ReLU activation replaces sigmoid.
- Dropout is added
- 2 fully connected layers are used instead of 3


To get the average performance of our model, we will train 20 of them with the same architecture and aggregate the results.

![Training size](https://imgur.com/9VUPYOp.jpg)
![CNN Output](https://imgur.com/vuN3vh9.jpg)

Our CNN model has an average accuracy of about 92% and an F1 of 89%.

Here is the confusion matrix for our average CNN model: 

![Confusion Matrix](https://imgur.com/AQ52ifVl.jpg)
