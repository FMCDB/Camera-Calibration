# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:32:25 2019

@author: Finian Bradwell (16306561)
"""

import numpy as np
import matplotlib.pyplot as plt


def calibrateCamera3D(data):
    '''Takes a matrix of correspondences and returns as input and returns 
    the perspective projection matrix of the camera as output''' 

    array1=np.zeros((982,12))
    
    #Fill initial matrix with correspondences
    
    '''
    Each correspondence results in 2 rows:
        ROW 1: X Y Z 1  0 0 0 0  -x*X -x*Y -x*Z -x
        ROW 2: 0 0 0 0  X Y Z 1  -y*X -y*Y -y*Z -y
    '''
    
    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]
    
    x = data[:,3]
    y = data[:,4]
    
    dataIndex = 0
    for i in range(0,len(array1),2):
        array1[i,0] = X[dataIndex]
        array1[i,1] = Y[dataIndex]
        array1[i,2] = Z[dataIndex]
        array1[i,3] = 1
        array1[i,8] = X[dataIndex]*-(x[dataIndex])
        array1[i,9] = Y[dataIndex]*-(x[dataIndex])  
        array1[i,10] = Z[dataIndex]*-(x[dataIndex])
        array1[i,11] = -(x[dataIndex])
        dataIndex += 1
        
    dataIndex = 0
    for i in range(1,len(array1),2):
        array1[i,4] = X[dataIndex]
        array1[i,5] = Y[dataIndex]
        array1[i,6] = Z[dataIndex]
        array1[i,7] = 1
        array1[i,8] = X[dataIndex]*-(y[dataIndex])
        array1[i,9] = Y[dataIndex]*-(y[dataIndex])
        array1[i,10] = Z[dataIndex]*-(y[dataIndex])
        array1[i,11] = -(y[dataIndex])
        dataIndex += 1
    
    #multiply the array by its transpose
    #generate the corresponding eigenvectors and eigenvalues
        
    D,V = np.linalg.eig(array1.transpose().dot(array1))
    
    #Find the position of the smallest eigenvalue
    
    array2 = V[:,np.argmin(D)]
        
    #Create the perspective projection matrix
        
    P = np.zeros((3,4))
    
    #Fill perspective projection matrix with eigenvectors
    
    for i in range(0,4):
        P[0,i] = array2[i]
    PColIndex = 0
    for i in range(4,8):
        P[1,PColIndex] = array2[i]
        PColIndex += 1
    PColIndex = 0
    for i in range(8,12):
        P[2,PColIndex] = array2[i]
        PColIndex += 1
        
    #print(P)
        
    return P


def visualiseCameraCalibration3D(data, P):
    ''' renders a single 2D plot showing (i) the measured 2D image point, and 
    (ii) the reprojection of the 3D calibration points as
    computed by P'''
    
    #Plot  measured 2D image data
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(data[:,3], data[:,4],'r.')
   
    #Store 3D data in a matrix
   
    data_3d = np.ones((491,4))
   
    for i in range(3):
        data_3d[:,i] = data[:,i]
        

    #Multiply 3D data by perspective projection matrix
    projected = P.dot(data_3d.transpose())

    #transpose it to make plotting easier
    projectedT = projected.transpose()
   
    #Rescale to fit dimensions 
    projectedT *= 255

    #plot reprojected 3D calibration points on 2d plane
    ax.plot(projectedT[:,0], projectedT[:,1], 'g.')
    plt.show()
   

def evaluateCameraCalibration3D(data, P):
    ''' print the mean, variance, minimum, and maximum
    distances in pixels between the measured and reprojected image feature 
    locations'''
        
    #Store 2D data in a matrix
    
    data_2d = np.ones((491,2))
   
    data_2d[:,0] = data[:,3]
    data_2d[:,1] = data[:,4]

    #Store 3D data in a matrix
   
    data_3d = np.ones((491,4))
   
    for i in range(3):
        data_3d[:,i] = data[:,i]

    #Multiply 3D data by perspective projection matrix
    projected = P.dot(data_3d.transpose())

    #transpose it to make plotting easier
    projectedT = projected.transpose()
   
    #Rescale to fit dimensions 
    projectedT *= 255
   
    #Remove third column from projectedT for computing distances
    projectedT_new = projectedT[:,:-1]
   
    #Print mean, variance, maximum and minimum distances between measured and reprojected points
    distances = np.ones((491))
    for i in range(len(data_2d)):
        distances[i] = np.linalg.norm(projectedT_new[i]-data_2d[i])
    print('Mean = ',np.mean(distances),
         '\nVariance = ',np.var(distances),
         '\nMinimum = ',np.amin(distances),
         '\nMaximum = ',np.amax(distances))
   
   
   
#Driver code to demonstrate operation of functions
        
data = np.loadtxt('data.txt')

P = calibrateCamera3D(data)

visualiseCameraCalibration3D(data,P)
    
evaluateCameraCalibration3D(data,P)


    
    



