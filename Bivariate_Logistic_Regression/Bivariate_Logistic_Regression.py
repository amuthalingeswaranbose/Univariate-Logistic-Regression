import numpy as np


class Bivariate_Logistic_Regression:
     
    def __init__(self, slope1=1, slope2=1, intercept=1, learning_rate=0.001, epochs=1000):
        
        # initialize
        self.slope1 = slope1
        self.slope2 = slope2
        #self.slope = np.ones(slope, dtype = int)
        self.intercept = intercept
        self.learning_rate = learning_rate
        self.epochs = epochs
      
    def fit(self, X_train, y_train):
        
        loss_history = []
        slope1_history = []
        slope2_history = []
        intercept_history = []
        
        for epoch in range(self.epochs):
            
            single_epoch_losses = []
            
            for x, y in zip(X_train, y_train):                
            
                # y = (m * x) + c - linear eqation
                #y_pred = (self.slope * x) + self.intercept 
                y_pred = (self.slope1 * x[0]) + (self.slope2 * x[1]) + self.intercept 
                
                # Sigmoid function
                sigmoid_of_y_pred = 1 / (1 + np.exp(-y_pred))   
                
                # calculate loss 
                loss = -(y*np.log(sigmoid_of_y_pred) + (1-y)*np.log(1-sigmoid_of_y_pred))
                
                # append single_epoch_losses loss
                single_epoch_losses.append(loss)
                
                # Find Derivatives of slope and intercept
                derivative_of_slope1 = ((sigmoid_of_y_pred - y)* x[0])
                derivative_of_slope2 = ((sigmoid_of_y_pred - y)* x[1])
                derivative_of_intercept = (sigmoid_of_y_pred - y)
                
                # Update slope and intercept
                self.slope1 -= self.learning_rate * derivative_of_slope1
                self.slope2 -= self.learning_rate * derivative_of_slope2
                self.intercept -= self.learning_rate * derivative_of_intercept
            
            average_epoch_loss = sum(single_epoch_losses) / len(X_train)
            print(f"iteration - {epoch} -> loss: {average_epoch_loss}, self.slope1: {self.slope1}, self.slope2: {self.slope2}, self.intercept: {self.intercept}")  
            
            loss_history.append(average_epoch_loss)
            slope1_history.append(self.slope1)
            slope2_history.append(self.slope2)
            intercept_history.append(self.intercept) 
            
        return self.slope1, self.slope1, self.intercept, loss_history, slope1_history, slope2_history, intercept_history
       
       
    def pred(self, X_test):
        
        y_hat_pred = []
        
        for xt in X_test:
                            
            y_hat = (self.slope1 * xt[0]) + (self.slope2 * xt[1]) + self.intercept
            sigmoid_of_y_hat = 1 / (1 + np.exp(-y_hat))
            
            if sigmoid_of_y_hat >= 0.5:   
            
                y_hat_pred.append(1)
                
            else:
            
                y_hat_pred.append(0)
        
        return y_hat_pred
        

    def pred_porb(self, X_test):
    
        y_hat_pred_prob = []
        
        for xt in X_test:
                            
            y_hat = (self.slope1 * xt[0]) + (self.slope2 * xt[1]) + self.intercept

            sigmoid_of_y_hat = 1 / (1 + np.exp(-y_hat))
            
            y_hat_pred_prob.append(sigmoid_of_y_hat)
        
        return y_hat_pred_prob
