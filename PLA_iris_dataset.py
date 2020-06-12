#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:42:45 2020
@author: gonzalobetancourt


PROGRAMMER: Gonzalo Betancourt 
COURSE: Intro To Machine Learning - 4336.01
DATE: February 19, 2020
ASSIGNMENT: Perceptron Model 
ENVIRONMENT: MacOS 
FILES INCLUDED: Source file PLA_iris_dataset.py

PURPOSE: 
        The purpose of this assignment was to learn in detail how to implement the 
        Perceptron algorithm. One of the main goals was to hard code the algorithm
        without using external libraries like Perceptron from sklearn.linear_model.
        Content:

        The Iris flower data set or Fisher's Iris data set is a multivariate 
        data set introduced by the British statistician and biologist Ronald 
        Fisher in his 1936 paper The use of multiple measurements in taxonomic 
        problems as an example of linear discriminant analysis.

        The data set consists of 50 samples from each of three species of 
        Iris (Iris setosa, Iris virginica and Iris versicolor). Four features 
        were measured from each sample: the length and the width of the sepals 
        and petals, in centimetres. Based on the combination of these four 
        features, Fisher developed a linear discriminant model to distinguish 
        the species from each other.

        features: in centimeters
        I used the Sepal width and Petal width for this project
         
INPUT: Inpouts will be features from a data set. For this project I used two features
       form the Iris Dataset, Sepal width and Petal width.
PRECONDITIONS: The initial weights of the algorithm will be initialized to 0.
        

OUTPUT: The weights and accuracy will be printed once the program is run 

POSTCONDITIONS: As a result the Perceptron algorithm will find the weights 
                that will linearly sperarate the data. 

ALGORITHM: 
          Calculate h using the dot product with weights ans x features
          update the weights using w = w0 + y*x[i]
            
EXAMPLE: 
        print("The accuracy of the model is: ", accuracy_score(y_test, y_pred))
        print("The final weights are: ", w)        
"""

import numpy as np
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions
from pandas import Series, DataFrame
from sklearn.metrics import accuracy_score


class Perceptron(object):
    """
    This class is meant to implement the Perceptron algorithm 
    """
    def __init__(self, sigma=1, epochs=10):
        """
        The perceptron will take in as paramerers.
        x_train
        y_train
        iterations: the defualt value is 10
        """
        self.epochs = epochs
        self.sigma = sigma # Learning Algorithm
        
    
    def train(self, x_train, y_train):
        """
        The function will use the data that was input when the Perceptron
        object was created. 
        
        X_trin and y_train
        return: the correct weights that will separate the linear data.
        """
        
        counter = 0
        
        self.w = np.zeros(len(x_train[0]))
        self.errors = np.ones(len(y_train))  # error vector
        # this vector is used for holding the estimated value during training
        self.y_est_vector = np.ones(len(y_train))
        self.sse = []  # vector for the SSE
        
        for i in range(self.epochs):
            for xi, target in zip(x_train, y_train):
                h_sign = self.w.dot(xi)  # sign of the output, xi is my features vector
                if h_sign >= 0:  # possitively classified
                    y_estimate = 1
                else:
                    y_estimate = 0  # negatively classified, I set 1 or 0 for my data
                self.y_est_vector[counter] = y_estimate
                
                # update weights after calculating h_sign. The weights may or may
                # not change depending on the value of h_sign
                for k in range(len(self.w)):
                    self.w[k] = self.w[k] + self.sigma * (target - y_estimate) * xi[k]
                
                counter += 1
            # calculating the errors
            for j in range(len(y_train)):
                self.errors[j] = (y_train[j] - self.y_est_vector[j]) ** 2
            self.sse.append(0.5*np.sum(self.errors))
            
            counter = 0
        return (self.w, self.sse)

    def test(self, x_test, y_test):
        """
        Function that will test the results using the returned weigts and the 
        test_data
        return the predicted values of the output
        """
        y_prediction = []
        for i in range(len(x_test)):
            h_prediction = self.w.dot(x_test[i])
            if h_prediction >= 0:
                y_estimate = 1
            else:
                y_estimate = 0 # I used 1 or 0 for my data instead of -1
                
            y_prediction.append(y_estimate)
        return y_prediction
    
    # These two functions make the graphing possible
    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def shuffle_data(dataset, data_points, random_seed = 5):
        """ 
        This static function will shuffle my data
        Parameters:
            Dataset
            how many data points
            default random_seed = 5 to shuffle the data
        """
        dataset = dataset.iloc[:data_points]
        dataset = dataset.values
        np.random.seed(random_seed)
        np.random.shuffle(dataset)
        # add 1 to the entire dataset as the first feature
        dataset = np.c_[np.ones(len(dataset)), dataset]
        return dataset 

#### function will return the testing and training data needed for PLA ###
def train_test_data_split(shuffled_data, grab_columns, percent_train=0.8, percent_test=0.2):
    """
    This static fucntion will return the training and testing sets
    Parameters:
        processed_data --> selected inouts and outputs including the amount of data points
        grab_columns --> the columns you want to include from the shuffled data
        percent_train --> 0 to 1 value: default is 0.8
        percent_test --> 0 to 1 value: default is 0.2
    return x_train, y_train, x_test, y_test
    """
    train = shuffled_data[:int(percent_train*len(shuffled_data)), grab_columns]
    test = shuffled_data[int(-percent_test*len(shuffled_data)):, grab_columns]
    
    x_train,y_train,x_test,y_test = train[:, :len(grab_columns) - 1], train[:, len(grab_columns) - 1], \
                                    test[:, :len(grab_columns) - 1], test[:, len(grab_columns) - 1]
    
    return (x_train, y_train, x_test, y_test)
    

if __name__ == "__main__":
    # importing the iris dataset.
    # I am only going to use two of the features,
    # Sepal width and Petal width
    # for the Setos and Versicolor
    iris = sklearn.datasets.load_iris()
    
    # prepocessing data so be used with the PLA model    
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_data['Class'] = iris.target
    
    # I will be using the first 100 data points from iris_data
    iris_test = shuffle_data(iris_data, 100)
    # I will input iris_test (shuffled data) and columns 0,2,4,5
    x_train, y_train, x_test, y_test = train_test_data_split(iris_test, [0, 1, 3, 5])
   
    iris_model = Perceptron()
    w, sse = iris_model.train(x_train, y_train)
    y_pred = iris_model.test(x_test, y_test)
    print("The accuracy of the model is: ", accuracy_score(y_test, y_pred))
    print("The final weights are: ", w)
    print("SSE Cost")
    print(sse)
    
    # Simple scatter plot that shows the linearly seperable data.
    plt.scatter(x_train[:,1], x_train[:,2], c = y_train,alpha=0.8) 
    plt.title("Perceptron")
    
    
    plot_decision_regions(x_train[:, 1:], y_train.astype(np.integer), clf=iris_model)
    plt.title('Perceptron Model')
    plt.xlabel('Sepal Width [cm]')   
    plt.ylabel('Petal Width [cm]')
    plt.show()
    
    
    plt.plot(range(1, len(sse) + 1), sse, marker='o')
    plt.xlabel('Iterations')   
    plt.ylabel('Misclassifiactions')
    plt.show()
    
    # Used the sklearn library to compare my results
    from sklearn.linear_model import Perceptron

    # training the sklearn Perceptron
    clf = Perceptron(random_state=None, eta0=1, shuffle=False, fit_intercept=False)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    
    print("The accuracy of the model is: ", accuracy_score(y_test, y_predict))
    print("The final weights are: ", clf.coef_)
    
