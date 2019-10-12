import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class NeuralNet:
    def __init__(self, url, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        
        raw_input = pd.read_csv(url,header=None)
        preprocessed_dataset = self.preprocess(raw_input)
        ncols = len(preprocessed_dataset.columns)
        nrows = len(preprocessed_dataset.index)
        x_data = preprocessed_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        y_data = preprocessed_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20)
        
        self.X = X_train
        self.y = y_train
        
        self.X_test = X_test
        self.y_test = y_test
        
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "tanh":
            self.__tanh(self,x)
        elif activation == "relu":
            self.__relu(self,x)

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "tanh":
            self.__tanh_derivative(self, x)
        elif activation == "relu":
            self.__relu_derivative(self, x)
            
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def __tanh(self,x):
        return (np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def __tanh_derivative(self,x):
        return (1 - np.square(x))
    
    def __relu(self,x):
        return np.maximum(x,0)
    
    def __relu_derivative(self,x):
        x[x<=0] = 0
        x[x>0]  = 1
        return x
    
    def removeNullValues(self,data):
        data_wo_nulls = data
        for i in range(len(data)):
            for j in range(len(data.iloc[i,:])):
                if(data.iloc[i,j] == '?'):
                    data_wo_nulls = data_wo_nulls.drop(data.index[i])
                    break
        return data_wo_nulls
    
    def normalizeData(self,data):
        mean = np.average(data)
        std_dev = np.std(data)
        normalizedData = (data - mean)/std_dev
        return normalizedData
    
    def standardizeData(self,data):
        standardizedData = data
        for i in range(len(data.columns)):
            data_type = data.dtypes[i]
            if( data_type == np.int64 or data_type == np.float64):
                standardizedData[i] = self.normalizeData(data[i])
        return standardizedData
    
    def encodeLabels(self,data):
        encodedData = data
        for i in range(len(encodedData.columns)):
            data_type = encodedData.dtypes[i]
            if(data_type == np.object):
                distinct_list = encodedData[i].unique().tolist()
                for j in range(len(encodedData)):
                    label = distinct_list.index(encodedData.iloc[j,i])
                    encodedData.iloc[j,i] = label
        return encodedData
    
    def classNormalization(self,data):
        index = len(data.columns)-1
        data[index]
        distinct_list = data[index].unique().tolist()
        
        labels_list = list(distinct_list)
        length = len(labels_list)
        interval = 1/length-1
        labels_list_dup = labels_list
        value = 0
        for i in range(len(labels_list)):
            labels_list_dup[i] = value
            value = value + interval
            
        labels_list_dup = np.array(labels_list_dup)
        
        for i in range(len(data)):
            value = data.iloc[i, index]
            label = labels_list_dup[distinct_list.index(value)]
            data.iloc[i, index] = label
        return data

    def preprocess(self, X):
        data_wo_nulls = self.removeNullValues(X)
        standardizedData = self.standardizeData(data_wo_nulls)
        encodedData = self.encodeLabels(standardizedData)
        preprocessed_data = self.classNormalization(encodedData)
        return pd.DataFrame(preprocessed_data)

    def train(self, max_iterations = 1000, learning_rate = 0.05, activation = "sigmoid"):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)
        
        return (np.sum(error)/len(error))

    def forward_pass(self,activation):
        if activation == "sigmoid":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        elif activation == "tanh":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        elif activation == "relu":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
            
        return out

    def backward_pass(self, out, activation):
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        elif activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    def predict(self, activation="sigmoid"):
        self.X = self.X_test
        self.y = self.y_test
        self.X01 = self.X
        out = self.forward_pass(activation)
        return ((np.sum(pow((self.y - out),2))) / (2*len(self.X)))

        
if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
   
    activation_func = ['sigmoid','tanh','relu']
    learning_rates = [0.0001,0.0005, 0.001,0.005,0.01,0.05,0.1,0.5]
    max_iterations = [100,250,500,750,1000,1500,2000,2500,5000]
    
    for activation in activation_func:
        for rate in learning_rates:
            for itr in max_iterations:
                neural_network = NeuralNet(url)
                trainError = neural_network.train(max_iterations=itr,learning_rate=rate,activation=activation)
                testError = neural_network.predict(activation=activation)
                dash = '-'*100
                print(dash)
                print("%-15s %-15s %-30s %-30s %s" %("Activation_Function","Learning_Rate","Maximum_Iterations","Training_Error","Testing_Error"))
                print(dash)
                print("%-19s %-18s %-30s %-30s %s" %(activation,str(rate),str(itr),str(trainError),str(testError)))
                print(dash)

