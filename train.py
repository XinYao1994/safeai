# 构建Safe AI：从零实现基于同态加密的神经网络
# http://iamtrask.github.io/2017/03/17/safe-ai/
# OpenMined上访问PySyft, 使用未加密的数据进行训练，但是神经网络模型在训练过程中完全加密的。
# 这导致了用户和具有智能的AI（简称智能AI）之间的权利失衡，这是非常有利的。
# 如果智能AI是同态加密的，那么从它的视角来看，整个外部世界也相当于是同态加密的（意思是不解密，不会互相造成影响）。
# 一个控制解密密钥的人，可以选择将智能AI模型本身解密，从而其释放到世界上，或仅将智能AI的某次单个预测结果解密，后者似乎更安全。




# 对数据进行加密后，虽然无法被读取，但是仍然保留统计学上的结构，
# 这使得人们可以在加密数据上训练模型(例如CryptoNets)
#%% Load data

from collections import Counter
import numpy as np
import time

path = "./"

labels = []
reviews = []

with open(path + "labels.txt") as f:
    labels = list(map(lambda x:x[:-1].upper(), f.readlines()))

with open(path + "reviews.txt") as f:
    reviews = list(map(lambda x:x[:-1], f.readlines()))

#%% 
# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, input_nodes, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        np.random.seed(1)
        self.init_network(input_nodes, hidden_nodes, 1, learning_rate)

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        # These are the weights between the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
        # These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        # The input layer, a two-dimensional matrix with shape 1 x hidden_nodes
        self.layer_1 = np.zeros((1, hidden_nodes))
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
    def train(self, reviews, labels):
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()
        
        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(reviews)):
            # Get the next review and its correct label
            review = reviews[i]
            label = labels[i]
            
            #### Implement the forward pass here ####
            ### Forward pass ###
            # Hidden layer
            ## New for Project 5: Add in only the weights for non-zero items
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]

            # Output layer
            ## New for Project 5: changed to use 'self.layer_1' instead of 'local layer_1'
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))            
            
            #### Implement the backward pass here ####
            ### Backward pass ###

            # Output error
            layer_2_error = layer_2 - label # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error

            # Update the weights
            ## New for Project 5: changed to use 'self.layer_1' instead of local 'layer_1'
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            
            ## New for Project 5: Only update the weights that were used in the forward pass
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate # update input-to-hidden weights with gradient descent step

            # Keep track of correct predictions.
            if(layer_2 >= 0.5 and label == 1):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 0):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            print("\rProgress:" + str(100 * i/float(len(reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        # keep track of how many correct predictions we make
        correct = 0
        # we'll time how many predictions per second we make
        start = time.time()
        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.reference(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            print("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def reference(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        self.layer_1 *= 0
        for index in review:
            self.layer_1 += self.weights_0_1[index]
        
        # Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        if(layer_2[0] >= 0.5):
            return 1
        else:
            return 0

class ReviewData():
    def __init__(self, reviews, labels) -> None:
        # populate review_vocab with all of the words in the given reviews
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        # populate label_vocab with all of the words in the given labels.
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def encodes(self, reviews, labels):
        x = []
        y = []
        for i in range(len(reviews)):
            x.append(list(self.encode_review(reviews[i])))
            y.append(self.get_target_for_label(labels[i]))
        return x, y

    def encode_review(self, review):
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        return unique_indices

    def get_target_for_label(self, label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0

#%%
reviewData = ReviewData(reviews[:-1000], labels[:-1000])
reviews_encode, labels_encode = reviewData.encodes(reviews[-1000:], labels[-1000:])

#%%
mlp = SentimentNetwork(reviewData.review_vocab_size, learning_rate=0.001)
mlp.test(reviews_encode, labels_encode)

#%%
mlp.train(reviews_encode, labels_encode)

#%%
mlp.test(reviews_encode, labels_encode)






























