import numpy as np
import random
import matplotlib.pyplot as plt

class Regression:
    def __init__(self):
        self.w = None
        self.b = None
    #data set
    def UniformRandomPoints(N,curve,a,b,AverageError):
        # generate points in the curve
        #.linspace creates an array of N evenly spaced numbers between a and b
        xval = np.linspace(a, b, N)
        #curve here is a function, 
        yval = [curve(k) for k in xval]
        # print('Display y:', yval)

        # add random numbers to simulate errors
        #enumerate fucntion here takes both idex and value of the list
        for i, value in enumerate(yval): 
            yval[i] = value + AverageError*random.uniform(-1, 1)

        result = []

        # calculate the result
        for yindex,yvalue in enumerate(yval):
            result.append([xval[yindex],yvalue])

        return result
    #gradient descent
    def train(self,x,y,alpha,iterations):
        # Number of training examples
        m = x.shape[0]    
        dj_dw = 0
        dj_db = 0
        self.w = 0
        self.b = 0
        
        # Gradient descent process
        for i in range(m):  
            f_wb = self.w * x[i] + self.b 
            dj_dw_i = (f_wb - y[i]) * x[i] 
            dj_db_i = f_wb - y[i] 
            dj_db += dj_db_i
            dj_dw += dj_dw_i 
        dj_dw = dj_dw / m 
        dj_db = dj_db / m 

        for i in range(iterations):
            tmp_w = self.w - alpha * dj_dw
            tmp_b = self.b - alpha * dj_db
            self.w = tmp_w
            self.b = tmp_b
            print("w: ",self.w," b: ",self.b)
        return self.w, self.b
    #cost fucntion
    def costFuction(self,x,y,w_trianed,b_trianed):
        print("w_trianed11: ",w_trianed," b_trianed11: ",b_trianed)
        m = x.shape[0] 
        cost_sum = 0 
        for i in range(m): 
            f_wb = w_trianed * x[i] + b_trianed   
            cost = (f_wb - y[i]) ** 2  
            cost_sum = cost_sum + cost  
        total_cost = (1 / (2 * m)) * cost_sum  

        return total_cost
    

data_set = Regression.UniformRandomPoints(4,lambda x: 2*x,0,4,0.1)
    
x = np.array([point[0] for point in data_set])
y = np.array([point[1] for point in data_set])

reg = Regression()
result =  reg.train(x,y,0.01,1000)

print("final result of training: ",result)

w_trianed = result[0]
b_trianed = result[1]

cost_function = reg.costFuction(x,y,w_trianed,b_trianed)
print("cost_function: ",cost_function)
            
        
