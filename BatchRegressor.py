import numpy as np
import math


class Regressor:
    def __init__(self,loss="mse",learning_rate=0.01):
        self.loss=loss
        self.lr=learning_rate
    def __set_weights_bias(self,n):
        self.features=int(n)
        self.weights = np.random.randn(n) * 0.01
        self.bias=0.0
    
    def __Mean_Squared_Error(self,Z,y):
        error=np.average(((Z-y)**2))/2
        return error
    
    def __adjust_weights(self,Z,X,y):
        error=Z-y
        db=np.mean(error)
        dw=np.mean(error.reshape(-1,1)*X,axis=0)
        self.bias-=(db*self.lr)
        self.weights-=(dw*self.lr)
            
    def __forwardProp(self,X):
        return ((X@self.weights)+self.bias)
    def __calc_error(self,Z,y):
        if self.loss=="mse":
            return self.__Mean_Squared_Error(Z,y)
    def fit(self,X,y,batch_size=1,epochs=0):
        self.__set_weights_bias(X.shape[1])
        y=y.flatten()
        for e in range(epochs):
            print(f"Epoch{e+1} : ------")
            self.__epoch(X,y,batch_size)
        

    def __epoch(self,X,y,batch_size):
        
        noOfBatches=math.ceil(X.shape[0]/batch_size)
        total_error=0
        total_samples=0
        for batches in range(noOfBatches):
            start_index=batches*batch_size
            end_index=(batches+1)*batch_size
            Z=self.__forwardProp(X[start_index:end_index])
            
            self.__adjust_weights(Z,X[start_index:end_index],y[start_index:end_index])
            total_error+=self.__calc_error(Z,y[start_index:end_index])*len(y[start_index:end_index])
            total_samples+=len(y[start_index:end_index])
        print("Error : ",total_error/total_samples)
        del X,y,batch_size
    
    def predict(self,X):
        return (X@self.weights)+self.bias
        


