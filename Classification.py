from sklearn import neural_network
import numpy as np

learning_rate = [0.1, 0.01]
max_nr_layer = 5;
max_hidden_layer = 5;

#the number instances read from file
data_size = 197                         #max 197 (raed "parkinsons.names" for more info)
#the number attributes
attributes = 23                         #max 23 (raed "parkinsons.names" for more info)

#splitting data into fit and train 
fit_data_count = int(0.75 * data_size)  #75% of instances are used for training
test_data_count = int(0.25 * data_size) #25% of instances are used for testing

data = np.zeros(shape = (data_size, 22)) #matrix of attributes
data_class = np.zeros(data_size)         #matrix of classes

model_nr = 0   
best_model_nr = 0   
best_model_accuracy = 0   




data_file = open("parkinsons.data", "r") #oppening data file
data_file.readline()                     #eliminating the first line wich contains the data instance description

#creating the results file
results = open("results.txt", "w")

for line_nr in range(0, data_size):
    
    #reading and splitting data
    data_str = data_file.readline()
    data_str = data_str.split(',')   
    
    #putting the data into an array
    line = np.zeros(attributes)
    for i in range(1, len(data_str)):
        
        line[i-1] = data_str[i]  
    
    #extracting the class from data
    class_ = line[16] #the lint 16 contains the class; 0 for healthy and 0 for PD (raed "parkinsons.names" for more info)
    line = np.delete(line, 16)
    
    #setting the maxtrices
    data[line_nr] = line
    data_class[line_nr] = class_
    
#closing data file
data_file.close()

#creating all posibble combination
for nr_layer in range(1,max_nr_layer):
    for hidden_layer in range(1, max_hidden_layer):    
        for lr in learning_rate:
            
            if nr_layer == 1:
                clf = neural_network.MLPClassifier(hidden_layer_sizes=(int(22*(1/hidden_layer))), learning_rate_init = lr)
            if nr_layer == 2:
                clf = neural_network.MLPClassifier(hidden_layer_sizes=(int(22*(1/hidden_layer)),int(22*(1/hidden_layer)*(1/hidden_layer))), learning_rate_init = lr)
            
            #creating matrices for training data
            fit_data = np.zeros(shape = (fit_data_count, 22))
            fit_data_class = np.zeros(fit_data_count)
            
            #setting the training data
            for n in range(0, fit_data_count):
                fit_data[n] = data[n]
                fit_data_class[n] = data_class[n]
            
            #training the model
            clf.fit(fit_data ,fit_data_class)
            
            #creating matrices for testing data
            test_data = np.zeros(shape = (test_data_count, 22))
            test_data_class = np.zeros(test_data_count)
            
            #setting the testing data
            for m in range(fit_data_count, fit_data_count+test_data_count):
                test_data[m-fit_data_count] = data[m-fit_data_count]
                test_data_class[m-fit_data_count] = data_class[m-fit_data_count]
            
            #predicting the class
            predict_class = clf.predict(test_data)
            
            #counting the correct prediction
            correct_prediction = 0
            for i in range(test_data_count):
                if test_data_class[i] == predict_class[i]:
                    correct_prediction += 1
                    
            accuracy = correct_prediction/test_data_count
            
            if accuracy > best_model_accuracy:
                best_model_accuracy = accuracy
                best_model_nr = model_nr
                
                    
            #printing the results
            results.write("Model " + str(model_nr) + ":\n" +
                  "\tNumber of hidden layers: " + str(nr_layer) + "\n" +
                  "\tNumber of neurons in the hidden layer: " + str(hidden_layer) + "\n" +
                  "\tLearning rate: " + str(lr) + "\n" +
                  "\tAccuracy: " + str(round(accuracy, 3)) + " (" + str(correct_prediction) + " out of " + str(test_data_count) + ")\n")
            
            model_nr += 1
            
results.write("\n\n----------\n The best model was " + str(best_model_nr) + " with an accuracy of " + str(round(best_model_accuracy, 3)));
            
results.close()