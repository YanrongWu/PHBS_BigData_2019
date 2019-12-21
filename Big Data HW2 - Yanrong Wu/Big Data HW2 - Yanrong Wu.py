#!/usr/bin/env python
# coding: utf-8

# # Homework 2
# ## Yanrong Wu 1801212952

# # Problem 1

# ## 1.Closed form function

# Q: Implement a function closed_form_1 that computes this closed form solution given the features 𝐗, labels Y (using Python or Matlab).

# In[51]:


import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt

climate_change_1 = pd.read_csv('climate_change_1.csv')
climate_change_1


# In[52]:


climate_change_1_train=climate_change_1.iloc[0:284]
#climate_change_1_train
climate_change_1_test=climate_change_1.iloc[284:308]
#climate_change_1_test


# In[53]:


def closed_form_1(df: pd.core.frame.DataFrame, column: int = 10)-> np.ndarray:
    X = df.drop(df.columns[column],axis=1).to_numpy()
    X = np.concatenate([np.ones((len(X),1)),X],axis = 1)
    # X: the features
    Y = df.iloc[:,[column]].to_numpy()
    Y = Y.reshape((len(Y)))
    # Y: the results
    theta = inv(X.T @ X)@ X.T @ Y
    return theta

def closed_form_2(X:np.ndarray, Y:np.ndarray)-> np.ndarray:
    theta = inv(X.T @ X)@ X.T @ Y
    return theta

closed_form_1(climate_change_1_train)[1:]


# In[54]:


# Using scipy to check:
from sklearn.linear_model import LinearRegression as lm
X=climate_change_1_train.drop(climate_change_1_train.columns[10],axis=1).to_numpy()
Y=climate_change_1_train.iloc[:,[10]].to_numpy()
l=lm().fit(X,Y)
l.coef_


# # 2.R2

# Q: Write down the mathematical formula for the linear model and evaluate the model R2 on the training set and the testing set.

# In[55]:


r_sq = l.score(X, Y)
r_sq


# # 3.Significant Variables

# Q: Which variables are significant in the model?

# In[56]:


import statsmodels.api as sm
mod = sm.OLS(Y,X)
fit = mod.fit()
p_values = fit.summary2().tables[1]['P>|t|']
p_values


# MEI,CO2,N2O,CFC-11,CFC-12,TSI,Aerosols are significant in the model(0.05 significant level).
# We can ignore the influence of the year and month later.

# # 4. For climate_change_2.csv

# Q: Write down the necessary conditions for using the closed form solution. And you can apply it to the dataset climate_change_2.csv, explain the solution is unreasonable.

# In[24]:


climate_change_2 = pd.read_csv('climate_change_2.csv')
#climate_change_2


# In[25]:


climate_change_2_train=climate_change_2.iloc[0:284]
climate_change_2_test=climate_change_2.iloc[284:308]
closed_form_1(climate_change_2_train)[1:]


# In[29]:


from sklearn.linear_model import LinearRegression as lm
X=climate_change_2_train.drop(climate_change_2_train.columns[10],axis=1).to_numpy()
Y=climate_change_2_train.iloc[:,[10]].to_numpy()
l=lm().fit(X,Y)
l.coef_


# In[28]:


import pandas as pd
climate_change_2_corr = climate_change_2.corr()
# Visualization
import matplotlib.pyplot as mp, seaborn
seaborn.heatmap(climate_change_2_corr, center=0, annot=True)
mp.show()


# It can be concluded from the correlation matrix that NO and CH4 are completely linearly correlated, so there is no inverse matrix, and the formula is invalid. So the solution is unreasonable.

# # Problem 2----Regularization

# # 1.Loss Function

# Q: Please write down the loss function for linear model with L1 regularization, L2 regularization, respectively.

# In[34]:


#L1 regularization
def L1Norm(l, theta):
    return  np.dot(np.abs(theta), np.ones(theta.size)) * l
 
def L1NormPartial(l, theta):
    return np.sign(theta) * l

# For linear regression, the derivative of J function is:
def __Jfunction(self):        
    sum = 0
    for i in range(0, self.m):
        err = self.__error_dist(self.x[i], self.y[i])
        sum += np.dot(err, err)
        sum += Regularization.L2Norm(0.8, self.theta)
        return 1/(2 * self.m) * sum


# In[37]:


#L2 regularization
def L2Norm(l, theta):
    return  np.dot(theta, theta) * l 
 
def L2NormPartial(l, theta):
    return theta * l

# For linear regression, the derivative of J function is:
def __partialderiv_J_func(self):
        sum = 0
        for i in range(0, self.m):
            err = self.__error_dist(self.x[i], self.y[i])
            sum += np.dot(self.x[i], err)
            sum += Regularization.L2NormPartial(0.8, self.theta)
            return 1/self.m * sum


# # 2.Closed Form Solution

# Q: The closed form solution for linear model with L2 regularization:
# 𝛉 = (𝐗𝐓𝐗 + 𝛌𝐈)−𝟏𝐗𝐓𝐘
# where I is the identity matrix. Write a function closed_form_2 that computes this closed form solution given the features X, labels Y and the regularization parameter λ.

# We can answer questions 2 and 4 together.

# In[48]:


def closed_form_2():

    dataset = pd.read_csv("climate_change_1.csv")
    X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])

    y = dataset.get("Temp")
  
    X = np.column_stack((X,np.ones(len(X))))

    for lambda1 in [10,1,0.1,0.01,0.001]:
        X_train = X[:284]
        X_test = X[284:]
        y_train = y[:284]
        y_test = y[284:]
    
        X_train=np.mat(X_train)
        y_train = np.mat(y_train).T
        xTx = X_train.T*X_train
        w = 0
        print("="*25+"L2 Regularization (lambda is "+str(lambda1)+")"+"="*25)
        I_m= np.eye(X_train.shape[1])
        if np.linalg.det(xTx+lambda1*I_m)==0.0:
            print("xTx is invertible")
        else:
            print(np.linalg.det(xTx+lambda1*I_m))
            w= (xTx+lambda1*I_m).I*(X_train.T*y_train)
        wights = np.ravel(w)    
        y_train_pred = np.ravel(np.mat(X_train)*np.mat(w))
        y_test_pred = np.ravel(np.mat(X_test)*np.mat(w))
        coef_=wights[:-1]
        intercept_=wights[-1]

        X_train = np.ravel(X_train).reshape(-1,9)
        y_train = np.ravel(y_train)
        
        print("Coefficient: ",coef_)
        print("Intercept: ",intercept_)
        print("the model is： y = ",coef_,"* X +(",intercept_,")")
        y_train_avg = np.average(y_train)
    
        R2_train = 1-(np.average((y_train-y_train_pred)**2))/(np.average((y_train-y_train_avg)**2))
        print("R2 in Train ： ",R2_train)
     
        y_test_avg = np.average(y_test)
        R2_test = 1-(np.average((y_test-y_test_pred)**2))/(np.average((y_test-y_test_avg)**2))
        print("R2 in Test ： ",R2_test)

closed_form_2()


# # 3.Comparasion

# Q: Compare the two solutions in problem 1 and problem 2 and explain the reason why linear model with L2 regularization is robust. (using climate_change_1.csv)

# It will reduce the coefficient of unimportant prediction factors close to 0 and avoid overfitting. In L2 model, it is less sensitive to single variable, so it is more robust.

# # 4.Change the regularization parameter λ

# Q: You can change the regularization parameter λ to get different solutions for this problem. Suppose we set λ = 10, 1, 0.1, 0.01, 0.001, and please evaluate the model R2 on the training set and the testing set. Finally, please decide the best regularization parameter λ. (Note that: As a qualified data analyst, you must know how to choose model parameters, please learn about cross validation methods.)

# The anwser can see the above(in Q2).

# # Problem 3 — Feature Selection

# # 1.Workflow

# Q: From Problem 1, you can know which variables are significant, therefore you can use less variables to train model. For example, remove highly correlated and redundant features. You can propose a workflow to select feature.

# Solution:
# For m features, from k=1 to k = m:
# We can choose k features from m features, and establish C (m, K) models, then choose the best one (MSE minimum or R2 maximum);
# Then select an optimal model from the m optimal models.

# # 2.Better Model

# Train a better model than the model in Problem 2.

# In[41]:


import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model

#Variance Inflation Factor
def vif(X, thres=10.0):
    col = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:,col].values, ix) for ix in range(X.iloc[:,col].shape[1])]
        maxvif = max(vif)
        maxix = vif.index(maxvif)
        if maxvif > thres:
            del col[maxix]
            print('delete=',X.columns[col[maxix]],'  ', 'vif=',maxvif )
            dropped = True
    print('Remain Variables:', list(X.columns[col]))
    print('VIF:', vif)
    return list(X.columns[col]) 

dataset = pd.read_csv("climate_change_1.csv")
X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])

y = dataset.get("Temp")

X_train = X[:284]
X_test = X[284:]
y_train = y[:284]
y_test = y[284:]
d = vif(X_train)
print(d)

X = dataset.get( ['MEI', 'CFC-12', 'Aerosols'])
y = dataset.get("Temp")
X_train = X[:284]
X_test = X[284:]
y_train = y[:284]
y_test = y[284:]

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
print('coefficients(b1,b2...):',regr.coef_)
print('intercept(b0):',regr.intercept_)
y_train_pred = regr.predict(X_train)
       
R2_1 = regr.score(X_train, y_train)
print(R2_1)
R2_2 = regr.score(X_test, y_test)
print(R2_2)


# # Problem 4 — Gradient Descent

# Gradient descent algorithm is an iterative process that takes us to the minimum of a function. Please write down the iterative expression for updating the solution of linear model and implement it using Python or Matlab in gradientDescent function.

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def costFunc(X,Y,theta):
    #cost func
    inner=np.power((X*theta.T)-Y,2)
    return np.sum(inner)/(2*len(X))

def gradientDescent(X,Y,theta,alpha,iters):
    temp = np.mat(np.zeros(theta.shape))
    cost = np.zeros(iters)
    thetaNums = int(theta.shape[1])
    
    for i in range(iters):
        error = (X*theta.T-Y)
        for j in range(thetaNums):
            derivativeInner = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j]-(alpha*np.sum(derivativeInner)/len(X))
        theta = temp
        cost[i]=costFunc(X,Y,theta)
    return theta,cost


dataset = pd.read_csv("climate_change_1.csv")
X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])

y = dataset.get("Temp")
X = np.column_stack((np.ones(len(X)),X))
X_train = X[:284]
X_test = X[284:]
y_train = y[:284]
y_test = y[284:]

X_train = np.mat(X_train)  
Y_train = np.mat(y_train).T

for i in range(1,9):
    X_train[:,i] = (X_train[:,i] - min(X_train[:,i])) / (max(X_train[:,i]) - min(X_train[:,i]))

theta_n = (X_train.T*X_train).I*X_train.T*Y_train
print("theta =",theta_n)
theta = np.mat([0,0,0,0,0,0,0,0,0])
iters = 100000
alpha = 0.001

finalTheta,cost = gradientDescent(X_train,Y_train,theta,alpha,iters)
print("final theta ",finalTheta)
print("cost ",cost)

fig, bx = plt.subplots(figsize=(8,6))
bx.plot(np.arange(iters), cost, 'r') 
bx.set_xlabel('Iterations') 
bx.set_ylabel('Cost') 
bx.set_title('Error vs. Training Epoch') 
plt.show()

