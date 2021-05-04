#!/usr/bin/env python
# coding: utf-8

# # Basic of Information Dynamics

# The following block aims to import all the relevant libraries to analyze data

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


# # Information content
# 
# function infocontent(p)
# Computes the Shannon information content for an outcome x of a random variable
# X with probability p.
# 
# Inputs:
# - p - probability to compute the Shannon info content for
# 
# Outputs:
# - result - Shannon info content of the probability p
#  
# Copyright (C) 2020, Joseph T. Lizier
# Distributed under GNU General Public License v3
# 

# In[2]:


def infocontent(p):    
    
    frac_logarithm = lambda x:np.log2(1/x)
    vfunc = np.vectorize(frac_logarithm)
    content = vfunc(p)
    
    return content

# For scalar p:
    # return np.log2(1/p)


# Comment for newbies**
#def frac_logarithm(x):
    
    #x = np.array(x)
    #log = np.log2(1/x)
    #return log
    
#def infocontent(p):
    
    #content = frac_logarithm(p)
    #return content


# # Entropy
# 
# function entropy(p)
# 
# Computes the Shannon entropy over all outcomes x of a random variable
# X with probability vector p(x) for each candidate outcome x.
# 
# Inputs:
# - p - probability distribution function over all outcomes x.
#        p is a vector, e.g. p = [0.25, 0.75], the sum over which must be 1.
# 
# Outputs:
# - result - Shannon entropy of the probability distribution p
#  
# Copyright (C) 2020, Joseph T. Lizier
# Distributed under GNU General Public License v3

# In[3]:


def entropy(p):  
    if type(p) == np.array:
        weightedShannonInfos = p*(infocontent(p))
    
    else:
        p = np.array(p)
        weightedShannonInfos = p*(infocontent(p))
    
    contributions = np.sum(weightedShannonInfos)
    
    return np.round(contributions,6)


# # Function jointentropy(p)
# 
# Computes the joint Shannon entropy over all outcome vectors x of a vector
# random variable X with probability matrix p(x) for each candidate outcome
# vector x.
# Inputs:
# - p - probability distribution function over all outcome vectors x.
#        p is a matrix over all combinations of the sub-variables of x,
# 	where p(1,3) gives the probability of the first symbol of sub-variable
# 	x1 co-occuring with the third symbol of sub-variable x2.
#        E.g. p = np.array([[0.2,0.3],[0.1,0.4]]). The sum over p must be 1.
# 
# Outputs:
# - result - joint Shannon entropy of the probability distribution p
#  
# Copyright (C) 2020, Joseph T. Lizier
# Distributed under GNU General Public License v3
# 

# In[4]:


def jointentropy(p):
    
    # Do I have to test whether this sums up 1?
    
    return entropy(p)


# # Function mutualinformation(p)
# 
# Computes the mutual information over all outcomes x of a random
# variable X with outcomes y of a random variable Y.
# Probability matrix p(x,y) is given for each candidate outcome
# (x,y).
# 
# Inputs:
# - p - 2D probability distribution function over all outcomes (x,y).
#     p is a matrix over all combinations of x and y,
#     where p(1,3) gives the probability of the first symbol of variable
#     x co-occuring with the third symbol of variable y.
#      E.g. p = np.array([[0.2, 0.3],[ 0.1, 0.4]]). The sum over p must be 1.
# 
# Outputs:
# - result - mutual information of X with Y
#  
# Copyright (C) 2017, Joseph T. Lizier
# Distributed under GNU General Public License v3
# 

# In[5]:


def mutualinformation(p):
    
    # We need to compute H(X) + H(Y) - H(X,Y):
    # 1. joint entropy:
    H_XY = jointentropy(p)

    # 2. marginal entropy of X:
    # But how to get p_x???
    p_x = p.sum(axis=1)  # Since x changes along the rows, summing /
                    # over the y's (dimension 2 argument in the sum) will just return p(x)
    
    H_X = entropy(p_x)

    # 2. marginal entropy of Y:
    # But how to get p_y???
    p_y = p.sum(axis=0); # Since y changes along the columns, summing over the x's (dimension 1 argument in the sum) will just return p(y)

    H_Y = entropy(p_y)
    
    mutualInfo = H_X + H_Y - H_XY
    
    return np.round(mutualInfo,5)


# # function conditionalentropy(p)
# 
# Computes the conditional Shannon entropy over all outcomes x of a random
# variable X, given outcomes y of a random variable Y.
# Probability matrix p(x,y) is given for each candidate outcome (x,y).
# 
# Inputs:
# - p - 2D probability distribution function over all outcomes (x,y).
#       p is a matrix over all combinations of x and y,
# where p(1,3) gives the probability of the first symbol of variable
# x co-occuring with the third symbol of variable y.
# E.g. p = np.array([[0.2, 0.3],[ 0.1, 0.4]]). The sum over p must be 1.
# 
# Outputs:
# - result - conditional Shannon entropy of X given Y
#  
# Copyright (C) 2020, Joseph T. Lizier
# Distributed under GNU General Public License v3

# In[6]:


def conditionalentropy(p):
    
    p = np.array(p)

    H_XY = jointentropy(p)
    
    p1, p2 = p.sum(axis=0) 
    p_primma = np.array([p1,p2])
    
    H_Y = entropy(p_primma)

    return H_XY-H_Y


# # Function conditionalmutualinformation(p)
# 
# Computes the mutual information over all outcomes x of a random
# variable X with outcomes y of a random variable Y, conditioning on 
# outcomes z of a random variable Z.
# Probability matrix p(x,y,z) is given for each candidate outcome
# (x,y,z).
# 
# Inputs:
# - p - 3D probability distribution function over all outcomes (x,y,z).
#        p is a matrix over all combinations of x and y and z,
# 	where p(1,3,2) gives the probability of the first symbol of variable
# 	x co-occuring with the third symbol of variable y and the second
# 	symbol of z.
#     The sum over p must be 1.
#     E.g.:
#         p(:,:,1) = [0.114286, 0.171429; 0.057143, 0.228571];
#         p(:,:,1) = [0.171429, 0.114286; 0.028571, 0.114286];
# 
# Outputs:
# - result - mutual information of X with Y
#  
# Copyright (C) 2020, Joseph T. Lizier
# Distributed under GNU General Public License v3
# 

# In[7]:


def conditionalmutualinformation(p):
    
    # 1. Joint Entropy
    H_XYZ = jointentropy(p)
    
    # 2. entropy of X,Z:
    # But how to get p_xz???
    # Sum p over the y's (dimension 2 argument in the sum) will just return p(x,z) terms. 
    # Won't be a 2D array, but fine to compute entropy on
    
    p_xz = p.sum(axis=1)
    H_XZ = jointentropy(p_xz)
    
    #3. entropy of Y,Z:
    # But how to get p_yz???
    # Sum p over the x's (dimension 1 argument in the sum) will just return p(y,z) terms. 
    # Won't be a 2D array, but fine to compute entropy on
    
    p_yz = p.sum(axis=0)
    H_YZ = jointentropy(p_yz)
    
    # 4. marginal entropy of Z:
    # But how to get p_z???
    # Sum p_xz over the x's (dimension 1 argument in the sum) will just return p(z) terms. 
    # Won't be a 1D array, but fine to compute entropy on
    
    p_z = p_xz.sum(axis=0)
    H_Z = jointentropy(p_z)
    
    # 5. Computing Conditional Mutual Information
    condMutInf = H_XZ - H_Z + H_YZ - H_XYZ
    
    return np.round(condMutInf, 6)


# # Plotting 

# In[8]:


def plotentropyX():
    
    alpha = np.arange(0,1,0.001)
    A = np.array([[1-alpha[0],alpha[0]/2],[0,alpha[0]/2]])
    
    for elem in alpha:
        A = np.append(A,np.array([[1-elem,elem/2],[0,elem/2]]),axis=0) 
    
    return A


# In[9]:


plotentropyX()


# # Function entropyempirical($x_{n}$)
# 
# Computes the Shannon entropy over all outcomes x of a random variable
# X from samples x_n.
# 
# Inputs:
# - xn - samples of outcomes x.
#       xn is a column vector, e.g. xn = [0;0;1;0;1;0;1;1;1;0] for a binary variable.
# 
# Outputs:
# - result - Shannon entropy over all outcomes
# 
# Copyright (C) 2020, Joseph T. Lizier
# Distributed under GNU General Public License v3
# 

# In[10]:


def entropyempirical(xn):
    
    # We need to work out the alphabet here.
    # The following returns a vector of the alphabet:
    
    if type(xn) == list:
        xn = np.array(xn)
    
    if xn.ndim == 1:
        xn = np.reshape(xn,(1,len(xn))) #reshaping our 1-dim vector to numpy format of a row vector
        xn = xn.T # Transposing the row vector to a column vector
    
    [c1,r1] = xn.shape 
    denominator = c1
    
    if c1 == 1:
        xn = xn.T
        [r1,c1] = xn.shape
        denominator = r1
        
    unique_rows = np.unique(xn, axis=0)
    probabilities = []
    for unique_row in unique_rows:
        count = 0
        for row in xn:
            if (row==unique_row).all():
                count += 1       
        probabilities.append(count/denominator)

    empEntropy = entropy(probabilities)
    return np.round(empEntropy,5)


# # Function jointentropyempirical(xn, yn)
# 
# Computes the Shannon entropy over all outcome vectors x of a vector random
# variable X from sample vectors x_n. User can call with two such arguments 
# if they don't wish to join them outside of the call.
# 
# Inputs:
# - xn - matrix of samples of outcomes x. May be a 1D vector of samples, or
#        a 2D matrix, where each row is a vector sample for a multivariate X.
# - yn - as per xn, except that yn is not required to be supplied (in which
#      case the entropy is only calculated over the xn variable).
# 
# Outputs:
# - result - joint Shannon entropy over all samples
#  
# Copyright (C) 2020, Joseph T. Lizier. Distributed under GNU General Public License v3

# In[11]:


def jointentropyempirical(xn, yn):
    
    xn = np.array(xn)
    yn = np.array(yn)
    
    if xn.ndim == 1:
        xn = np.reshape(xn,(1,len(xn)))
    
    if yn.ndim == 1:
        yn = np.reshape(yn,(1,len(yn)))
    
    [c1,r1] = xn.shape
    [c2,r2] = yn.shape
    
    if c1 == 1 and c2 == 1:
        XY = np.concatenate((xn,yn), axis=0)
        jointEntrEmp = entropyempirical(XY.T)
        return jointEntrEmp
    
    elif c1 != c2:
        XY = np.row_stack((xn,yn))
        jointEntrEmp = entropyempirical(XY.T)
        return jointEntrEmp
    
    else:
        XY = np.concatenate((xn,yn),axis=1)
        jointEntrEmp = entropyempirical(XY)
        return jointEntrEmp


def jointentropyempiricalMV(xn):
    
    jointEntrEmp = entropyempirical(xn)
    return jointEntrEmp


# # Function mutualinformationempirical($x_{n},y_{n}$)
# 
# Computes the mutual information over all samples $x_{n}$ of a random
# variable $X$ with samples $y_{n}$ of a random variable $Y$.
# 
# Inputs:
# - xn - matrix of samples of outcomes x. May be a 1D vector of samples, or
#        a 2D matrix, where each row is a vector sample for a multivariate X.
# - yn - matrix of samples of outcomes x. May be a 1D vector of samples, or
#        a 2D matrix, where each row is a vector sample for a multivariate Y.
#        Must have the same number of rows as X.
# 
# Outputs:
# - result - mutual information of X with Y
#  
# Copyright (C) 2020, Joseph T. Lizier. Distributed under GNU General Public License v3

# In[12]:


def mutualinformationempirical(xn,yn):
    # We need to compute H(X) + H(Y) - H(X,Y):
    # 1. joint entropy:
    H_XY = jointentropyempirical(xn, yn)
    # 2. marginal entropy of Y: (calling 'joint' in case yn is multivariate)
    H_Y = entropyempirical(yn)
    #3. marginal entropy of X: (calling 'joint' in case yn is multivariate)
    H_X = entropyempirical(xn)

    result = H_X + H_Y - H_XY
    
    return result
    


# # Function conditionalentropyempirical(xn,yn)
# 
# Computes the conditional Shannon entropy over all samples xn of a random
# variable X, given samples yn of a random variable Y.
# 
# Inputs:
# - xn - matrix of samples of outcomes x. May be a 1D vector of samples, or
#        a 2D matrix, where each row is a vector sample for a multivariate X.
# - yn - matrix of samples of outcomes x. May be a 1D vector of samples, or
#        a 2D matrix, where each row is a vector sample for a multivariate Y.
#        Must have the same number of rows as X.
# 
# Outputs:
# - result - conditional Shannon entropy of X given Y
#  
# Copyright (C) 2020, Joseph T. Lizier. Distributed under GNU General Public License v3

# In[13]:


def conditionalentropyempirical(xn,yn):
    
    # We need to compute H(X,Y) - H(X):
    # 1. joint entropy:
    H_XY = jointentropyempirical(xn, yn)
    # 2. marginal entropy of Y: (calling 'joint' in case yn is multivariate)
    H_Y = jointentropyempiricalMV(yn)

    result = H_XY - H_Y
    
    return result
    
    


# # Function conditionalmutualinformationempirical(xn,yn)
# 
# Computes the mutual information over all samples xn of a random
# variable X with samples yn of a random variable Y, conditioning on 
# samples zn of a random variable Z.
# 
# Inputs:
# - xn - matrix of samples of outcomes x. May be a 1D vector of samples, or
#        a 2D matrix, where each row is a vector sample for a multivariate X.
# - yn - matrix of samples of outcomes y. May be a 1D vector of samples, or
#        a 2D matrix, where each row is a vector sample for a multivariate Y.
#        Must have the same number of rows as X.
# - zn - matrix of samples of outcomes z. May be a 1D vector of samples, or
#        a 2D matrix, where each row is a vector sample for a multivariate Z
#        which will be conditioned on.
#        Must have the same number of rows as X.
# 
# Outputs:
# - result - conditional mutual information of X with Y, given Z
#  
# Copyright (C) 2020, Joseph T. Lizier. Distributed under GNU General Public License v3

# In[14]:


def conditionalmutualinformationempirical(xn,yn,zn):
    """    # Should we check any potential error conditions on the input?
    if (isvector(xn)):
        # Convert it to column vector if not already:
        if (size(xn,1) == 1):
            # xn has only one row:
            xn = xn.T # Transpose it so it is only column
    
    if (isvector(yn)):
        # Convert it to column vector if not already:
        if (size(yn,1) == 1):
            # yn has only one row:
            yn = yn.T # Transpose it so it is only column

    if (isvector(zn)):
        # Convert it to column vector if not already:
        if (size(zn,1) == 1):
            # zn has only one row:
            zn = zn.T # Transpose it so it is only column

    # Check that their number of rows are the same:
    assert(size(xn,1) == size(yn,1))
    assert(size(xn,1) == size(zn,1))
    """
    
    # Concatenate xn and yn to compute conditional entropy empircal
    [c1,r1] = xn.shape
    [c2,r2] = yn.shape
    
    if c1 == 1 and c2 == 1:
        xy = np.concatenate((xn,yn), axis=0)   
    else:
        xy = np.concatenate((xn,yn),axis=1)


    # We need to compute H(X|Z) + H(Y|Z) - H(X,Y|Z):
    # 1. conditional joint entropy:
    H_XY_given_Z = conditionalentropyempirical(xy, zn)
    # 2. conditional entropy of Y:
    H_Y_given_Z = conditionalentropyempirical(yn, zn)
    # 3. conditional entropy of X:
    H_X_given_Z = conditionalentropyempirical(xn, zn)

    result = H_X_given_Z + H_Y_given_Z - H_XY_given_Z

    # Alternatively, note that we could compute I(X;Y,Z) - I(X;Z)
    # 1. joint MI:
    # I_X_YZ = mutualinformationempirical(xn, [yn, zn])
    # 2. MI just from Z:
    # I_X_Z = mutualinformationempirical(xn, zn)
    # Then:
    # result = I_X_YZ - I_X_Z
    
    return result

