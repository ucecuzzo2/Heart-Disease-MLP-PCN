import numpy as np

#For our activation functiuon
def sigmoid_act(a):
    return 1 / (1 + np.exp(-a)) #Sigmoid forumula

#For out derivatve on sigmoid
def sigmoid_derivative(a):
    return a * (1 - a) #Sigmoid derivative forumula

