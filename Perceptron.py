import numpy as np

class PCN: #define the perceptron class
    #Define initslization of self and operations(number of operations in which we will use pereprton on used to determine size of vector)
    def __init__ (self,operations):
         # Initalize weights and bias
         self.weights = np.zeros(operations)
         self.bias = 0
    
    def Predict(self, x):
         # Performs binary classifications on linear model based on the dot product of 
         #input x and the set of weights which is added to a bias term and checks if
         # that value is either > 0 or < 0 if true return 1(class) else return 0 (class)
         return np.where(np.dot(x,self.weights) + self.bias >= 0,1,0)
         
    def train_compute(self,x,y,rate_learning = 0.1,epochs = 100):
         #Assign learning rate (determines how much should be updated on the slope from parametrs)
         # and epochs (complete pass through training data set) so it has to be set to 100 in this assignment
         #rate_learning = 0.1
         #epochs = 100
         for _ in range(epochs): #iterate through 100 
              for i in range(len(x)): # within inner loop iterate through each training daats set (x)
                  predict = self.Predict(x[i]) # instantiate Precict on each x iteration assign to predict
                  update = rate_learning * (y[i] - predict) # used to compute the product of rate_learning 
                  # and difference of y[i] to help learn from any mistakes to get better predictions
                  self.weights += update * x[i] # update weights based on x[i]
                  self.bias += update #update bias with update
                  # These two our updates to our perceptron algorithm to better improe accuracy
