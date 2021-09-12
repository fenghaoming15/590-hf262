# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 22:11:43 2021

@author: Haoming Feng
"""

#--------------------------------
# UNIVARIABLE REGRESSION EXAMPLE
#    -USING SciPy FOR OPTIMIZATION
#--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize


#------------------------
#CODE PARAMETERS
#------------------------

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"


OPT_ALGO='BFGS'    #HYPER-PARAM

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type="logistic"; NFIT=4; X_KEYS=['x']; Y_KEYS=['y']
# model_type="linear";   NFIT=2; X_KEYS=['x']; Y_KEYS=['y']
# model_type="logistic"; NFIT=4; X_KEYS=['y']; Y_KEYS=['is_adult']

#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
p=np.random.uniform(0.5,1.,size=NFIT)

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]

#------------------------
#DATA CLASS
#------------------------

class DataClass:

    #INITIALIZE
    def __init__(self,FILE_NAME):

        if(FILE_TYPE=="json"):

            #READ FILE
            with open(FILE_NAME) as f:
                self.input = json.load(f)  #read into dictionary

            #CONVERT DICTIONARY INPUT AND OUTPUT MATRICES #SIMILAR TO PANDAS DF   
            X=[]; Y=[]
            for key in self.input.keys():
                if(key in X_KEYS): X.append(self.input[key])
                if(key in Y_KEYS): Y.append(self.input[key])

            #MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
            self.X=np.transpose(np.array(X))
            self.Y=np.transpose(np.array(Y))
            self.been_partitioned=False

            #INITIALIZE FOR LATER
            self.YPRED_T=1; self.YPRED_V=1

            #EXTRACT AGE<18
            if(model_type=="linear"):
                self.Y=self.Y[self.X[:]<18]; 
                self.X=self.X[self.X[:]<18]; 

            #TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
            self.XMEAN=np.mean(self.X,axis=0); self.XSTD=np.std(self.X,axis=0) 
            self.YMEAN=np.mean(self.Y,axis=0); self.YSTD=np.std(self.Y,axis=0) 
        else:
            raise ValueError("REQUESTED FILE-FORMAT NOT CODED"); 

    def report(self):
        print("--------DATA REPORT--------")
        print("X shape:",self.X.shape)
        print("X means:",self.XMEAN)
        print("X stds:" ,self.XSTD)
        print("Y shape:",self.Y.shape)
        print("Y means:",self.YMEAN)
        print("Y stds:" ,self.YSTD)

    def partition(self,f_train=0.8, f_val=0.15,f_test=0.05):
        #TRAINING:      DATA THE OPTIMIZER "SEES"
        #VALIDATION: NOT TRAINED ON BUT MONITORED DURING TRAINING
        #TEST:         NOT MONITORED DURING TRAINING (ONLY USED AT VERY END)


        if(f_train+f_val+f_test != 1.0):
            raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

        #PARTITION DATA
        rand_indices = np.random.permutation(self.X.shape[0])
        CUT1=int(f_train*self.X.shape[0]); 
        CUT2=int((f_train+f_val)*self.X.shape[0]); 
        self.train_idx, self.val_idx, self.test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
        self.been_partitioned=True

    def model(self,x,p):
        if(model_type=="linear"):   return  p[0]*x+p[1]  
        if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.0001))))

    def predict(self):
        self.YPRED_T=self.model(self.X[self.train_idx],self.p)
        self.YPRED_V=self.model(self.X[self.val_idx],self.p)
        self.YPRED_TEST=self.model(self.X[self.test_idx],self.p)
        

    def normalize(self):
        self.X=(self.X-self.XMEAN)/self.XSTD 
        self.Y=(self.Y-self.YMEAN)/self.YSTD  

    def un_normalize(self):
        self.X=self.XSTD*self.X+self.XMEAN 
        self.Y=self.YSTD*self.Y+self.YMEAN 
        self.YPRED_V=self.YSTD*self.YPRED_V+self.YMEAN 
        self.YPRED_T=self.YSTD*self.YPRED_T+self.YMEAN 
        self.YPRED_TEST=self.YSTD*self.YPRED_TEST+self.YMEAN 

    #------------------------
    #DEFINE LOSS FUNCTION
    #------------------------
    def loss(self,p):
        er = self.model(self.X[self.train_idx],p) - self.Y[self.train_idx]
        return np.mean((er**2))
    
    def loss_validation(self,p):
        er = self.model(self.X[self.val_idx],p) - self.Y[self.val_idx]
        return np.mean((er**2))
    
    def loss_mifirst(self,p):
        index_50 = self.train_idx[:len(self.train_idx)//2]
        er = self.model(self.X[index_50],p) - self.Y[index_50]
        return np.mean((er**2))
    
    def loss_milast(self,p):
        index_last50 = self.train_idx[len(self.train_idx)//2 + 1 :]
        er = self.model(self.X[index_last50],p) - self.Y[index_last50]
        return np.mean((er**2))
            

    def fit(self):
        #TRAIN MODEL USING SCIPY MINIMIZ 
        res = self.optimize("GD", 0.001,method = "batch")
        popt=res; print("OPTIMAL PARAM:",popt)
        self.p = popt

        #PLOT TRAINING AND VALIDATION LOSS AT END
        if(IPLOT):
            fig, ax = plt.subplots()
            ax.plot(iterations, loss_train, 'o', label='Training loss')
            ax.plot(iterations, loss_val, 'o', label='Validation loss')
            plt.xlabel('optimizer iterations', fontsize=18)
            plt.ylabel('loss', fontsize=18)
            plt.legend()
            plt.show()

    #FUNCTION PLOTS
    def plot_1(self,xla='x',yla='y'):
        if(IPLOT):
            fig, ax = plt.subplots()
            ax.plot(self.X[self.train_idx]    , self.Y[self.train_idx],'o', label='Training') 
            ax.plot(self.X[self.val_idx]      , self.Y[self.val_idx],'x', label='Validation') 
            ax.plot(self.X[self.test_idx]     , self.Y[self.test_idx],'*', label='Test') 
            ax.plot(self.X[self.train_idx]    , self.YPRED_T,'.', label='Model') 
            plt.xlabel(xla, fontsize=18);    plt.ylabel(yla, fontsize=18);     plt.legend()
            plt.show()

    #PARITY PLOT
    def plot_2(self,xla='y_data',yla='y_predict'):
        if(IPLOT):
            fig, ax = plt.subplots()
            ax.plot(self.Y[self.train_idx]  , self.YPRED_T,'*', label='Training') 
            ax.plot(self.Y[self.val_idx]    , self.YPRED_V,'*', label='Validation') 
            ax.plot(self.Y[self.test_idx]    , self.YPRED_TEST,'*', label='Test') 
            plt.xlabel(xla, fontsize=18);    plt.ylabel(yla, fontsize=18);     plt.legend()
            plt.show()
            
    def plot_3(self,xla ='epoch times',yla = 'loss'):
        times = np.arange(50000)
        if(IPLOT):
            fig,ax = plt.subplots()
            ax.plot(times,self.losslist,label = "training_loss")
            ax.plot(times,self.lossval,label = "validation_loss")
            plt.xlabel(xla, fontsize=18);    plt.ylabel(yla, fontsize=18);     plt.legend()
            plt.show()
            
    #optimize from scratch       
    def optimize(self,algo,LR,method):
        if algo == "GD":
            t = 0
            t_max = 50000
            dp = 0.0001
            p = np.random.uniform(0.5,1.,size=4)
            self.losslist = []
            self.lossval = []
            if method == "batch":
                while t < t_max:
                    
                    dl_dp1 = (self.loss([p[0]+dp,p[1],p[2],p[3]]) - self.loss(p))/dp
                    dl_dp2 = (self.loss([p[0],p[1]+dp,p[2],p[3]]) - self.loss(p))/dp
                    dl_dp3 = (self.loss([p[0],p[1],p[2]+dp,p[3]]) - self.loss(p))/dp
                    dl_dp4 = (self.loss([p[0],p[1],p[2],p[3]+dp]) - self.loss(p))/dp
                    self.losslist.append(self.loss(p))
                    self.lossval.append(self.loss_validation(p))
                    
                    
                    
                    grad_p = [dl_dp1 * LR,dl_dp2 * LR,dl_dp3 * LR,dl_dp4 * LR]
                
                    p = p - grad_p
                    t += 1
            if method == "mini_batch":
                while t < t_max:
                    #First 50%
                    dl_dp1 = (self.loss_mifirst([p[0]+dp,p[1],p[2],p[3]]) - self.loss_mifirst(p))/dp
                    dl_dp2 = (self.loss_mifirst([p[0],p[1]+dp,p[2],p[3]]) - self.loss_mifirst(p))/dp
                    dl_dp3 = (self.loss_mifirst([p[0],p[1],p[2]+dp,p[3]]) - self.loss_mifirst(p))/dp
                    dl_dp4 = (self.loss_mifirst([p[0],p[1],p[2],p[3]+dp]) - self.loss_mifirst(p))/dp
                
                    grad_p = [dl_dp1 * LR,dl_dp2 * LR,dl_dp3 * LR,dl_dp4 * LR]
                
                    p = p - grad_p
                    
                    #Last 50%
                    dl_dp1 = (self.loss_milast([p[0]+dp,p[1],p[2],p[3]]) - self.loss_milast(p))/dp
                    dl_dp2 = (self.loss_milast([p[0],p[1]+dp,p[2],p[3]]) - self.loss_milast(p))/dp
                    dl_dp3 = (self.loss_milast([p[0],p[1],p[2]+dp,p[3]]) - self.loss_milast(p))/dp
                    dl_dp4 = (self.loss_milast([p[0],p[1],p[2],p[3]+dp]) - self.loss_milast(p))/dp
                    
                    grad_p = [dl_dp1 * LR,dl_dp2 * LR,dl_dp3 * LR,dl_dp4 * LR]
                
                    p = p - grad_p
                    
                    self.losslist.append(self.loss(p))
                    self.lossval.append(self.loss_validation(p))
                    t += 1
            
            if method == "stochastic":
                while t < t_max:
                    np.random.shuffle(self.train_idx)
                    
                    for data in self.train_idx:
                        
                        loss = np.mean((self.model(self.X[data],p) - self.Y[data])**2)

                
                        dl_dp1 = (np.mean((self.model(self.X[data],[p[0]+dp,p[1],p[2],p[3]]) - self.Y[data]))**2- loss)/dp
                        dl_dp2 = (np.mean((self.model(self.X[data],[p[0],p[1]+dp,p[2],p[3]]) - self.Y[data]))**2- loss)/dp
                        dl_dp3 = (np.mean((self.model(self.X[data],[p[0],p[1],p[2]+dp,p[3]]) - self.Y[data]))**2- loss)/dp
                        dl_dp4 = (np.mean((self.model(self.X[data],[p[0],p[1],p[2],p[3]+dp]) - self.Y[data]))**2- loss)/dp
                        
                        grad_p = [dl_dp1 * LR,dl_dp2 * LR,dl_dp3 * LR,dl_dp4 * LR]                                 
                        
                        p = p - grad_p
                    self.losslist.append(self.loss(p))
                    self.lossval.append(self.loss_validation(p))
                    if t %1000 == 0:
                        print(t)
                        print(dl_dp1)
                        print(dl_dp2)
                        print(dl_dp3)
                        print(dl_dp4)
                        print(grad_p)
                        print(p)
                    t += 1
                    
                    
            return p
                
                
        


#------------------------
#MAIN 
#------------------------
D=DataClass(INPUT_FILE)        #INITIALIZE DATA OBJECT 
D.report()                    #BASIC DATA PRESCREENING

D.partition()                #SPLIT DATA
D.normalize()                #NORMALIZE
D.fit()
D.predict()
D.plot_1()
D.plot_2()
D.plot_3()

