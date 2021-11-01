# Useful starting lines
import numpy as np
from proj1_helpers import *

'''Table of content:
Section 1 [l.25 to l.45] : Small helper functions for mandatory functions
Section 2 [l.57 to l.120] : Mandatory functions
Section 3 [l.122 to end] : Feature engineering functions
                   -decorrelate_features
                   -delete_zero_var_features
                   -delete_outliers
                   -add_offset
                   -standardize
                   -divide_in_groups
                   -polynomial_embedding
                   -cross_validation
                   -build_k_indices
                   -cross_validation_demo
                   -split_data
                   -bias_variance_demo
                   -cross_validation_deg_lambda
                   -cross_validation_bis
'''

'''Small helper functions for mandatory functions==========================================='''
#Loss function (MSE)
def MSE_loss(y, tx, w):
    x=y-tx.dot(w)
    return 1/(2*len(y))*x.dot(x)

#Gradient of loss function (MSE_loss)
def Grad(y, tx, w):
    x=y-tx.dot(w)
    z=tx.T.dot(x)
    return -(1/len(y))*z 

#applies sigmoid function for log reg
def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

#Computes the log_loss for log reg using sigmoid (negative likellyhood)
def log_loss(y, tx, w):
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

#Stochastic gradient descent
def SGD(y, tx, w):
    my=None
    mx=None
    for mmini_y,mini_x in batch_iter(y, tx, batch_size=1):
            my=mmini_y
            mx=mini_x
    
    return Grad(my, mx, w)

'''Mandatory functions=================================================================='''
#Least squares regression thanks to normal equations
def least_squares(y, tx):
    weight = np.linalg.solve(tx.T@tx,tx.T@y)
    return weight, MSE_loss(y,tx,weight)

#Least squares using GD
def least_squares_GD(y, tx, initial_w,max_iters, gamma):
    weight=initial_w
    for n_iter in range(max_iters):
        weight=weight-gamma*Grad(y,tx,weight)
        
        if n_iter%20==0:
            loss=MSE_loss(y,tx,weight)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=weight[0], w1=weight[1]))
            
    loss=MSE_loss(y,tx,weight)
    return weight,loss

#Least square using SGD
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    weight=initial_w
    for i in range(max_iters):
        grd=gamma*SGD(y,tx,weight)
        weight=weight-grd
    loss=MSE_loss(y,tx,weight)
    
    return weight, loss
#Ridge regression
def ridge_regression(y, tx, lambda_):
    weight = np.linalg.solve(tx.T@tx + lambda_*np.eye(tx.shape[1]),tx.T@y)
    return weight, MSE_loss(y,tx,weight)

#logistic regression with GD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    #updating the weights by finding a minina using GD
    weight = initial_w
    for i in range(max_iters):
        loss = log_loss(y, tx, weight)
        grad = Grad(y, tx, weight)
        weight -= gamma * grad
        
        if n_iter%20==0:
            loss=log_loss(y,tx,weight)
            print("Gradient Descent : iter =", inter," loss =",loss,"")
            
    return weight, loss

def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    #regularized log reg with GD
    
    weight = initial_w
    
    for i in range(max_iters) :
        loss = log_loss(y, tx, weight) + lambda_ * np.squeeze(weight.T.dot(weight))
        gradient = Grad(y, tx, weight) + 2 * lambda_ * weight
        w -= gamma *gradient
        
        if n_iter%20==0:
            loss=log_loss(y,tx,weight)
            print("Gradient Descent : iter =", inter," loss =",loss,"")
        
    return weight, loss

'''Feature engineering functions====================================================='''

#Function to delete highly correlated features : "decorrelate features"
def decorrelate_features(x, threshold=.9):
    corr = np.corrcoef(np.transpose(x))
    row = np.expand_dims(np.where(corr >= threshold)[0],axis=1)
    col = np.expand_dims(np.where(corr >= threshold)[1],axis=1)
    indices = np.concatenate([row, col], axis=1)
    indices = indices[indices[:,0] < indices[:, 1]]

    to_delete = np.array([], dtype = np.int32)
    for i in range(indices.shape[0]):
        if not(indices[i, 1] in to_delete):
            index = int(indices[i, 1])
            to_delete = np.concatenate([to_delete, np.array([index])])
    to_delete.astype(np.int64)
    return np.delete(x, to_delete, axis=1), to_delete

#Function to delete features with zero variance
def delete_zero_var_features(x):
    var = np.var(x, axis=0)
    to_keep = np.nonzero(var)
    return np.squeeze(x[:, to_keep], axis=1)

#Delete outliers that are above a certain treshold (personalized for individual features)
def delete_outliers(x, threshold):
    bool_vect = np.all(x>threshold, axis=1)
    legit_ids = np.where(bool_vect)
    clean_x = x[legit_ids, :]
    return clean_x, legit_ids

#Add offset 
def add_offset(tX):
    return np.concatenate([np.ones((tX.shape[0], 1)), tX], axis=1)

#Standardize the data
def standardize(x):
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data

#Function to divide the data into 6 groups
def divide_in_groups(x):
    
    group00 = np.where(x[:, 0]==-999)[0]
    group01 = np.where(x[:, 0]!=-999)[0]
    group1 = np.where(x[:, 22]==0)[0]
    group2 = np.where(x[:, 22]==1)[0]
    group3 = np.where(x[:, 22]>1)[0]
    
    ids_01 = np.intersect1d(group00, group1)
    ids_02 = np.intersect1d(group00, group2)
    ids_03 = np.intersect1d(group00, group3)
    ids_11 = np.intersect1d(group01, group1)
    ids_12 = np.intersect1d(group01, group2)
    ids_13 = np.intersect1d(group01, group3)
    ids = (ids_01, ids_02, ids_03, ids_11, ids_12, ids_13)
    
    x01 = x[ids_01, :]
    x02 = x[ids_02, :]
    x03 = x[ids_03, :]
    x11 = x[ids_11, :]
    x12 = x[ids_12, :]
    x13 = x[ids_13, :]
    
    delete_from_gr1 = [4, 5, 6, 12, 23, 24, 25, 26, 27, 28]
    delete_from_gr2 = [4, 5, 6, 12, 25, 26, 27, 28]
    
    x01 = np.delete(x01, delete_from_gr1, axis=1)
    x11 = np.delete(x11, delete_from_gr1, axis=1)
    x02 = np.delete(x02, delete_from_gr2, axis=1)
    x12 = np.delete(x12, delete_from_gr2, axis=1)
    x01 = np.delete(x01, 0, axis=1)
    x02 = np.delete(x02, 0, axis=1)
    x03 = np.delete(x03, 0, axis=1)
    x13 = np.delete(x13,22,axis=1)
    
    data = (x01, x02, x03, x11, x12, x13)
    
    return data, ids

#Feature augmentation with polynomial embedding
def polynomial_embedding(x, degree=2):
    if degree==0 or degree==1 :
        return x
    res = x
    for i in range(1, degree):
        pows = (i+1)*np.ones((x.shape[1],))
        res = np.concatenate((res, np.power(x, pows)), axis=1)
    return res


def cross_validation(y, x, k_indices, k, lambda_):
    # ***************************************************
    index = k_indices[k-1,:]
    k_indices = np.delete(k_indices,(k-1),axis=0)
    k_indices = np.reshape(k_indices, k_indices.size)
    x_test = x[index-1]
    y_test = y[index-1]
    x_train = x[k_indices-1]
    y_train = y[k_indices-1]
    # get k'th subgroup in test, others in train
    # ***************************************************
    w,loss_tr = ridge_regression(y_train,x_train,lambda_)
    # ridge regression
    # ***************************************************
    loss_te = MSE_loss(y_test,x_test,w)
    # calculate the loss for train and test data
    # ***************************************************
    return loss_tr, loss_te

#building the indices for k fold cross validation
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

#cross_validation
def cross_validation_demo(y,x, seed=1):
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    for j in range(30):
        sum_tr = 0
        sum_te = 0
        for i in range(k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, i, lambdas[j])
            sum_tr = sum_tr + np.sqrt(2*loss_tr)
            sum_te = sum_te + np.sqrt(2*loss_te)
            
        rmse_tr = np.append(rmse_tr, sum_tr/k_fold) #Take the mean of the values so we divide by 4
        rmse_te = np.append(rmse_te, sum_te/k_fold)
    # cross validation: TODO
    # *************************************************** 
    index = np.where(rmse_te == np.min(rmse_te))
    print(index)
    return lambdas[index]


#splitting the data into a training et test set, according to the ratio
def split_data(x, y, ratio, seed=1):

    # set seed
    np.random.seed(seed)
    # ***************************************************
    n = len(y)
    indices = np.random.permutation(n)
    index_split = int(np.floor(ratio * n))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    
    x_train = x[index_tr]
    x_test = x[index_te]
    y_train = y[index_tr]
    y_test = y[index_te]
    # ***************************************************
    return x_train, x_test, y_train, y_test


#cross validation of the degree
def bias_variance_demo(y,tx,lambda_):
    # define parameters
    seeds = range(50)
    #print(seeds)
    ratio_train = 0.5
    degrees = range(1, 10)
    
    # define list to store the variable
    rmse_tr = np.empty((len(seeds), len(degrees)))
    rmse_te = np.empty((len(seeds), len(degrees)))
    
    for index_seed, seed in enumerate(seeds):
        np.random.seed(seed)
        # ***************************************************
        x_train, x_test, y_train, y_test = split_data(tx, y, ratio_train, seed)
        # split data with a specific seed
        # ***************************************************
        for i in degrees:
            tx_train = polynomial_embedding(x_train,i)
            tx_test = polynomial_embedding(x_test,i)
            w,loss_tr = ridge_regression(y_train,tx_train,lambda_)
            loss_te = MSE_loss(y_test,tx_test,w)
            rmse_tr[index_seed,i-1] = np.sqrt(loss_tr*2)
            rmse_te[index_seed,i-1] = np.sqrt(loss_te*2)
        # bias_variance_decomposition
        # *************************************************
        
    index = np.where(rmse_te == np.min(rmse_te))[1]
    #print(index)
    #print(degrees)
    return degrees[index[0]]

def cross_validation_deg_lambda(y,tx):
    seed = 50
    ratio_train = 0.5
    degrees = range(1,10)
    
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define list to store the variable
    rmse_tr = np.empty((len(lambdas), len(degrees)))
    rmse_te = np.empty((len(lambdas), len(degrees)))
    print(rmse_tr.shape)
    print(rmse_te.shape)
    
    for j in range(30):
        np.random.seed(seed)
        # ***************************************************
        #x_train, x_test, y_train, y_test = split_data(tx, y, ratio_train, seed)
        # split data with a specific seed
        # ***************************************************
        for i in degrees:
            sum_tr = 0
            sum_te = 0
        
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation_bis(y, tx, k_indices, k, lambdas[j],i)
                sum_tr = sum_tr + np.sqrt(2*loss_tr)
                sum_te = sum_te + np.sqrt(2*loss_te)
            
            rmse_tr[j,i-1] = sum_tr/k_fold #Take the mean of the values so we divide by 4
            rmse_te[j,i-1] = sum_te/k_fold
        # bias_variance_decomposition
        # *************************************************

    index0 = np.where(rmse_te == np.min(rmse_te))[0]
    index1 = np.where(rmse_te == np.min(rmse_te))[1]
    print(index0)
    print(index1)
    return lambdas[index0[0]], degrees[index1[0]]

#Cross validation, returns the loss of ridge regression
def cross_validation_bis(y, x, k_indices, k, lambda_,i):
    # ***************************************************
    index = k_indices[k-1,:]
    k_indices = np.delete(k_indices,(k-1),axis=0)
    k_indices = np.reshape(k_indices, k_indices.size)
    x_test = x[index-1]
    y_test = y[index-1]
    x_train = x[k_indices-1]
    y_train = y[k_indices-1]
    # get k'th subgroup in test, others in train
    # ***************************************************
    x_train = polynomial_embedding(x_train,i)
    x_test = polynomial_embedding(x_test,i)
    w,loss_tr = ridge_regression(y_train,x_train,lambda_)
    # ridge regression
    # ***************************************************
    loss_te = MSE_loss(y_test,x_test,w)
    # calculate the loss for train and test data
    # ***************************************************
    return loss_tr, loss_te