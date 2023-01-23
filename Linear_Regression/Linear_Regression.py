import numpy as np

class Linear_Regression:
     
    def __init__(self, slope=1, intercept=1, learning_rate=0.001, epochs=1000):
        
        # initialize
        self.slope = slope
        self.intercept = intercept
        self.learning_rate = learning_rate
        self.epochs = epochs
      
    def fit(self, X_train, y_train):
        
        loss_history = []
        slope_history = []
        intercept_history = []
        
        for epoch in range(self.epochs):
            
            single_epoch_losses = []
            
            for x, y in zip(X_train, y_train):                
            
                # y = (m * x) + c - linear eqation
                y_pred = (self.slope * x) + self.intercept 
                
                # calculate loss - MSE - Mean Squard Error
                diff = (y_pred - y)
                loss = diff**2
                #print(f"loss: {loss}")
                
                # append single_epoch_losses loss
                single_epoch_losses.append(loss)
                
                # Find Derivatives of slope and intercept
                derivative_of_slope = 2*(y_pred - y) * x
                derivative_of_intercept = 2*(y_pred - y)
                
                # Update slope and intercept
                self.slope -= self.learning_rate * derivative_of_slope
                self.intercept -= self.learning_rate * derivative_of_intercept
            
            average_epoch_loss = sum(single_epoch_losses) / len(X_train)
            print(f"iteration - {epoch} -> loss: {average_epoch_loss}, self.slope: {self.slope}, self.intercept: {self.intercept}")  
            
            loss_history.append(average_epoch_loss)
            slope_history.append(self.slope)
            intercept_history.append(self.intercept) 
            
        return self.slope, self.intercept, loss_history, slope_history, intercept_history  

    def pred(self, X_test):
    
        y_hat_pred = []
        
        for xt in X_test:
                            
            y_hat = (self.slope * xt) + self.intercept
            
            y_hat_pred.append(y_hat)
        
        return y_hat_pred
