#1)Your program will first read the above file for the Heart Disease patients.  
import numpy as np # Use for numpy array
import csv # for file open
from Perceptron import PCN # Import my PCN implmentation 
from Helper_Function import sigmoid_act, sigmoid_derivative # Import our activation and  sigmoid functions 


with open("heart2.csv",'r') as file_h:
        csvreader = csv.reader(file_h)
        #Delcare our data set as a list
        heart_data = []
        for row in csvreader: # for each row in the csvreader
            try: # try append val floats for each val in the row on csv reader
                  heart_data.append([float(val) for val in row])
            except ValueError: # will will continue if there is non-numeric values such as age, gender...etc
                  continue
        #Using numpy we can use this into a array for numpy
        heart_data = np.array(heart_data) # assigned heart_data into numpy array        
        #Step 2 

iterations = int(input("How many times do you want to apply the Perceptron and MLP network ? "))

count = 0
#Iniatlize our data sets for raw and preprocessed for PCN
preprocesses_data_pcn = []
raw_data_pcn = []

##Iniatlize our data sets for raw and preprocessed for MLP
preprocesses_data_MLP = []
raw_data_MLP = []


patience = 10  # Adjust this value as needed
best_pre_accur_mlp = 0
epoch_best = 0 # for epoch on early stopping

#Assign our lengths and sizes for MLP
size_output = 1 #define output level
size_hidden_layer = 100 #hidden layer size 100
size_input = len(heart_data[0]) -1 # size of input layer subtact 1 from number of features in heart

#For loop assigned to our iteartions
for i in range (iterations): 
      
      #Random shuffle compute
      np.random.shuffle(heart_data)

      #Separate a quarter of the data for training and the other quarter for testing,
      # By assigning our boudnaries for testing and trainng data
      training_size = int (0.25 *len(heart_data)) # Calculatye size of train set quarter
      training_data = heart_data[:training_size] #Create training set
      testing_data = heart_data[training_size:] #CReate testing set

      #PCN Initialization
      pcn = PCN (operations = len(training_data[0])-1) #Initial instance of PCN
      x_train = training_data[:,:-1] # assign our x on trainng
      y_train = training_data[:,-1] # assign our y on training
      pcn.train_compute (x_train,y_train) # train our x and y

      #Apply PCN for preprocessing:
      #subtact min value by scale of range and sub min value for preprocvessing on training data
      train_pre_data = (training_data - np.min(training_data, axis = 0) / np.max(training_data, axis = 0) - np.min(training_data, axis = 0)) # Process pre-processed data
      #Similar for preprocess but for testing 
      test_pro_data = (testing_data - np.min(training_data, axis = 0) / np.max(training_data, axis = 0) - np.min(training_data, axis = 0)) # Process processed test data

      #Grab input feature and label from target from pree processed data training
      pre_x_train_pcn = train_pre_data [:,:-1]
      pre_y_train_pcn = train_pre_data [:, -1]

      #Initalize PCN for preprocessed data
      pcn_pre = PCN(operations = len(train_pre_data[0]) - 1 )
      pcn_pre.train_compute(pre_x_train_pcn,pre_y_train_pcn) #Train Preprocess data
      test_pcn_x_pre = test_pro_data[:,:-1] #Grab all input feature from test_pcn_y_pre
      test_pcn_y_pre = test_pro_data[:,-1] ##Grab all input feature from test_pcn_y_pre

      #Now we calcualte the accuracies of our PCN
      pcn_acc_raw = np.mean(pcn.Predict(testing_data[:,:-1]) == testing_data[:,-1])
      pcn_acc_preprocessed = np.mean(pcn_pre.Predict(test_pcn_x_pre) == test_pcn_y_pre) 

      #Iniatalize our MLP for data and implementation
      mlp_size_train = int(0.50 * len(heart_data)) #Calculate size of MLP train whihch is half
      mlp_size_valid = int(0.25 * len(heart_data)) #Calculate size of MLP train whihch is quarter
      training_data_mlp = heart_data [:mlp_size_train] #Creqate training data from the rows of mlp_size_train
      validation_data_mlp = heart_data [mlp_size_train:mlp_size_train +mlp_size_valid] #Create validatio data mlp from rows mlp_size_train to mlp_size_valid 
      testing_data_mlp =  heart_data [mlp_size_train + mlp_size_valid:] #Create testing data for rows mlp_size_train and mlp_size_valid

      # MLP apply normilaization:
      val_max = np.max(training_data_mlp, axis = 0)  #Calculate the max of training with respect to our axis
      val_min = np.min(training_data_mlp, axis = 0) #Calcuale the min of training with respetc to our axis
      
      for index_feature in range(size_input): # Used for overload issues wince we can compute this part seperately instead in one go
            min_val = val_min[index_feature]
            max_val = val_max[index_feature]
            training_data_mlp[:,index_feature] = (training_data_mlp[:,index_feature] - min_val) / (max_val-min_val)  #Normialize our data by subtact train - min / max -min
            validation_data_mlp [:,index_feature] = (validation_data_mlp[:,index_feature] - min_val) / (max_val-min_val)  #Normalize our data by sub valid - min  max - min
            testing_data_mlp [:,index_feature] = (testing_data_mlp[:,index_feature] - min_val) / (max_val-min_val) #Normailize our test - val / max - min

            # training_data_mlp = (training_data_mlp - val_min) / (val_max - val_min) #Normialize our data by subtact train - min / max -min
            # validation_data_mlp = (validation_data_mlp - val_min) / (val_max - val_min) #Normalize our data by sub valid - min  max - min
            # testing_data_mlp = (testing_data_mlp - val_min) / (val_max - val_min) #Normailize our test - val / max - min

      #Extract our inputs from x train and y train frrom our training mlp data 
      x_train_mlp, y_train_mlp = training_data_mlp [:,:-1] , training_data_mlp [:,- 1]
      #Extract our inputs of x an y validation from our validation set
      x_val_mlp, y_val_mlp = validation_data_mlp[:,:-1], validation_data_mlp[:,-1]
      #Extract our x and y test from our testing set
      x_test_mlp, test_y_mlp = testing_data_mlp[:,:-1], testing_data_mlp[:,-1]

      #Assign our weights and bias for MLP hidden inputs and outpust
      input_hidden_weights = 2 * np.random.rand(size_input,size_hidden_layer) -1
      output_hidden_weights = 2 * np.random.rand(size_hidden_layer,size_output) -1
      #Initalzie random weight for input and hidden layer creates matrix of these weights

      

      #Multi-Level Perceptron (MLP) for a fixed number of iterations (say, several hundred or thousand 
      for epoch in range(1000):
        
        

        #for our layers (FowardPropagation)
        input_hidden = np.dot(x_train_mlp,input_hidden_weights) #Calcuate the weighted sum of hidden layer
        output_hidden = sigmoid_act(input_hidden) # Calculate sigmoid activation function for hidden layer sum output
        output_input_layer = np.dot(output_hidden,output_hidden_weights) #Calculate weighted sum of inputs of output layer
        output_output_layer = sigmoid_act(output_input_layer) #Apply sigmoid on output layers

        # (Back Propagation) Note-> d output variable and rest of d are denoted as derivative
        error_back = y_train_mlp.reshape(-1,1) - output_output_layer #perfom error calculaytion of target table
        d_output = error_back * sigmoid_derivative(output_output_layer) #Calculate derivative of error by applyoh sigmoid_derivative
        error_back_hidden = d_output.dot (output_hidden_weights.T) #Calculaye backpropagtion of error on the output layer
        d_hidden = error_back_hidden * sigmoid_derivative(output_hidden) #Calculate derivative of hidden layer using sigmoid derivative

        #Adjust our bias and weights of MLP Note-> T (is our transpose operation for our martix)
        output_hidden_weights += output_hidden.T.dot(d_output) #Adjust weights by adding out hidden layer output 
        input_hidden_weights += x_train_mlp.T.dot(d_hidden) #Adjust weights for input layer 

      #Perfom calculation of MLP raw data
        mlp_accuracy_raw = np.mean(np.round(output_output_layer) == y_train_mlp) #Perform our eound on MLP output againts  our y_train_mlp for accuracy 
        #Perform calculation on preprocessed MLP data
        input_hidden = np.dot(x_test_mlp,input_hidden_weights) # Calculate wighted sum  of input_hidden _weights
        output_hidden = sigmoid_act(input_hidden) #Calculate our sigmoid with input_hidden
        output_input_layer = np.dot(output_hidden,output_hidden_weights) #Calculate wighted sum of hidden layers
        output_output_layer = sigmoid_act(output_input_layer) # Pass again to activation functpon used for our final calculation of accuracy for preprocessed
        mlp_accuracy_pre = np.mean(np.round(output_output_layer) == test_y_mlp)

        #Early Stopping applied.
        if mlp_accuracy_pre > best_pre_accur_mlp : # check is current mlp accuracy is our best one
            best_pre_accur_mlp = mlp_accuracy_pre # if it is assign both accordinly
            best_epoch = epoch # upate best_epoch to the current

        if  epoch - best_epoch > patience: # else no improvemnnt then we apply early stopping
            break


      #Append our calc to list
      raw_data_pcn.append(pcn_acc_raw)
      preprocesses_data_pcn.append(pcn_acc_preprocessed)
      preprocesses_data_MLP.append(mlp_accuracy_pre)
      raw_data_MLP.append(mlp_accuracy_raw)

      # count += 1
      # print ("Iteartions:", count)


#Finally we can calculate our averae acuracies MLP and PCN
a_pcn_raw = np.mean(raw_data_pcn)
a_pcn_pre = np.mean(preprocesses_data_pcn)
a_mlp_raw = np.mean(raw_data_MLP)
a_mlp_pre = np.mean(preprocesses_data_MLP)

#Print out average accuracies
print("The Perceptron average accuracy of:", iterations, "experiments on raw data is:", a_pcn_raw)
print("The Perceptron average accuracy of:", iterations, "experiments on preprocessed data is:", a_pcn_pre)
print("")
print("The MLP average accuracy of:", iterations, "experiments on raw data is:", a_mlp_raw)
print("The MLP average accuracy of:", iterations, "experiments on preprocessed data is:", a_mlp_pre)













     
















      
      