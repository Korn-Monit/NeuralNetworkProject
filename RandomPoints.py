
import numpy as np
import random
import matplotlib.pyplot as plt

# N: number of points
# a,b: range in which we will generate points
# AverageError: the average distance between the gererated points and the curve
def UniformRandomPoints(N,curve,a,b,AverageError):
    # generate points in the curve
    xval = np.linspace(a, b, N)
    yval = [curve(k) for k in xval]

    # add random numbers to simulate errors
    for i, value in enumerate(yval): 
        yval[i] = value + AverageError*random.uniform(-1, 1)

    result = []

    # calculate the result
    for yindex,yvalue in enumerate(yval):
        result.append([xval[yindex],yvalue])

    return result

def GaussRandomPoints(N,curve,a,b,sigma):
    # generate points in the curve
    xval = np.linspace(a, b, N)
    yval = [curve(k) for k in xval]

    # add random numbers to simulate errors
    for i, value in enumerate(yval): 
        yval[i] = value + random.gauss(0, sigma)

    result = []

    # calculate the result
    for yindex,yvalue in enumerate(yval):
        result.append([xval[yindex],yvalue])

    return result




# Testing functions

def LineairFunction(x):
    return x**2

def ShowGaussRandomPoints():
    points = GaussRandomPoints(100,LineairFunction,10,0,5)

    # Extract x and y coordinates into separate lists
    x = [coord[0] for coord in points]
    y = [coord[1] for coord in points]

    plt.scatter(x, y)
    plt.show()

def ShowLineairRandomPoints():
    points = UniformRandomPoints(100,LineairFunction,10,0,5)

    # Extract x and y coordinates into separate lists
    x = [coord[0] for coord in points]
    y = [coord[1] for coord in points]

    plt.scatter(x, y)
    plt.show()

