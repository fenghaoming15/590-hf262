#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 00:02:38 2021

@author: haomingfeng
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statistics
from statistics import mean
import matplotlib.pyplot as plt
from scipy.optimize import fmin, minimize, fmin_tnc
import math

class Data:
    def __init__(self,path):
        self.data = pd.read_json(path)
        
    
    def partition_linear(self):
        feature_column = self.data["x"]
        label_column = self.data["y"]
        x_train, x_test, y_train, y_test = train_test_split(feature_column,label_column,test_size = 0.3)
        return x_train,x_test,y_train,y_test
    
    def normalize(self):
        x_mean = mean(self.data["x"])
        y_mean = mean(self.data["y"])
        x_std = statistics.pstdev(self.data["x"])
        y_std = statistics.pstdev(self.data["y"])
        self.data["x"] = [(i-x_mean)/x_std for i in self.data['x']]
        self.data["y"] = [(i-y_mean)/y_std for i in self.data['y']]
    
    def visualize_linear(self):
        plt.scatter(self.data["x"],self.data["y"])
        plt.show()
    
    def return_data(self):
        return self.data
    
    

mydata = Data('weight.json')
x_train,x_test,y_train,y_test = mydata.partition_linear()
under_18 = x_train <= 18
under_18_test = x_test <= 18
x_train = x_train[under_18]
y_train = y_train[under_18]
x_test = x_test[under_18_test]
y_test = y_test[under_18_test]

#Linear Regression
p0 = np.zeros([x_train.shape[0], 1])

p1 = np.zeros([x_train.shape[0],1])

def pred(x, para):
    return np.dot(x, para[0]) + para[1]

y_pred = pred(x_train, [0,0])

def loss(para):
    p = pred(x_train, para)
    e = y_train - p
    se = np.power(e, 2)
    rse = np.sqrt(np.sum(se))
    rmse = rse / y_train.shape[0]
    return rmse

min = fmin(loss, [0,0], maxiter=1000)

y_min = pred(x_train, min)

out = pd.DataFrame({'y': y_train, 'y_pred': y_pred, 'y_min': pred(x_train, min)})
out.head(n=15)

#use minimize()
#nms = minimize(loss, w, method='nelder-mead')

#make another prediction
#out_2 = pd.DataFrame({'y': y[:,0], 'y_pred': y_pred[:,0], 'y_min': pred(x, nms.x)})
#out_2.head()
fig,ax = plt.subplots()
ax.plot(x_train,y_train,"o",label = "train_data")
ax.plot(x_test,y_test,"x",color = "y",label = "test_data")
ax.plot(x_train,x_train*min[0]+min[1])
leg = ax.legend()
plt.xlabel("Age")
plt.ylabel("Weight")
plt.show()

#Logistic regression
x_train_log,x_test_log,y_train_log,y_test_log = mydata.partition_linear()


def pred_logistic(x,para):
    return para[0]+para[1]*(1.0/(1.0+np.exp(-(x-para[2])/para[3])))


def loss_logistic(para):
    p = pred_logistic(x_train_log, para)
    e = y_train_log - p
    se = np.power(e, 2)
    rse = np.sqrt(np.sum(se))
    rmse = rse / y_train.shape[0]
    return rmse

min = fmin(loss_logistic,[0,0,0,0])

#y_min = pred(x_train, min)

#out = pd.DataFrame({'y': y_train, 'y_pred': y_pred, 'y_min': pred(x_train, min)})
#out.head(n=15)

#use minimize()
#nms = minimize(loss_logistic, [0,0,0,0], method='nelder-mead')


#def model(x,p0,p1,p2,p3):
#    return p0+p1*(1.0/(1.0+np.exp(-(x-p2)/(p3+0.00001))))
fig,ax = plt.subplots()
ax.plot(x_train_log,y_train_log,"o",label = "train_data")
ax.plot(x_test_log,y_test_log,"x",color = "y",label = "test_data")
#ax.plot(x_train,min[0] + min[1]*(1/(1+math.exp(-(x_train-min[2]/min[3]+0.00001)))))
ax.plot(x_train_log,pred_logistic(x_train_log,min))
leg = ax.legend()
plt.xlabel("Age")
plt.ylabel("Weight")
plt.show()

#Logistic Regression for classification
#Reference:https://towardsdatascience.com/logistic-regression-with-python-using-optimization-function-91bd2aee79b
mydata.normalize()
label = mydata.return_data()["is_adult"]
weight = mydata.return_data()["y"]
weight = weight.values.reshape(-1,1)
label = label.values.reshape(-1,1)

def sigmoid(x,theta):
    z = np.dot(x,theta)
    return 1/(1+np.exp(-z))

def hypothesis(theta,x):
    return sigmoid(x,theta)

def cost_function(theta,x,y):
    m = weight.shape[0]
    h = hypothesis(theta,x)
    return -(1/m)*np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient(theta,x,y):
    m = weight.shape[0]
    h = hypothesis(theta, x)
    return (1/m) * np.dot(weight.T, (h-y))

theta = np.zeros((weight.shape[1], 1))

def fit(x,y,theta):
    opt_weights = fmin_tnc(func = cost_function,x0 = theta,fprime=gradient,args = (x,y.flatten()))
    return opt_weights[0]
parameters = fit(weight,label,theta)
pred_y = hypothesis(parameters,weight)

fig,ax = plt.subplots()
ax.plot(weight,label,"o",label = "train_data")
ax.plot(weight,pred_y,label = "logistic regression")
leg = ax.legend()
plt.xlabel("weight")
plt.ylabel("adult")
plt.show()







        
    