#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:42:45 2020

@author: gonzalobetancourt

The following program is the Preceptron Learning Algorithm.
To understand how the model works. First I will code the example I saw in class. 

Using two data points each with only one feature. I will then move on to a larger real world data set. 

"""
import numpy as np
import pandas as pd


data = pd.read_csv("two_points_example.csv", keep_default_na=False)


x_vals = data.iloc[:,0]
y_vals = data.iloc[:, 1]


w = np.array([0, 0])
x_1 = np.array([1, x_vals[0]])
x_2 = np.array([1, x_vals[1]])
x_data = np.array([[1, -0.3],[1, 3.0]])

h1_x1 = 0
h2_x2 = 0
i = 0
for item in x_data:
    h = w.dot(item)
    if not(h >= 0):
        w = w + y_vals[i] * x_data[i]
    if not(h < 0):
        w = w + y_vals[i] * x_data[i]
    i = i + 1

while not(h1_x1 < 0) and not(h2_x2 > 0):
    # look for correct classification
    h1_x1 = w.dot(x_1)
    h2_x2 = w.dot(x_2)
    if not(h1_x1 >= 0):
        w = w + y_vals[0] * x_1
    if not(h2_x2 < 0):
        w = w + y_vals[1] * x_2

print("The final weights are: ", w)
print("The final h1(x): ", h1_x1)
print("The final h2(x): ", h2_x2)



"""
x1 = np.array([-0.3, 1])
x2 = np.array([3, -1])

# initialze the weights to zero so as not lose generalization
w = np.array([0, 0])

# h_1_x = w*x (scalar prodcut)

x_1 = np.array([1, -0.3])
x_2 = np.array([1, 3])

h1_x1 = 0
h2_x2 = 0

while not(h1_x1 >= 0) and not(h2_x2 < 0):
    # look for correct classification
    h1_x1 = w.dot(x_1)
    h2_x2 = w.dot(x_2)
    
    if not(h1_x1 >= 0):
        w = w + x1[1] * x_1
    if not(h2_x2 < 0):
        w = w + x2[1] * x_2

print("The final weights are: ", w)
print("The final h1(x): ", h1_x1)
print("The final h2(x): ", h2_x2)
"""


