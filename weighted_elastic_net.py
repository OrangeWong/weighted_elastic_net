# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:59:23 2018

@author: richa
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

class WEN(BaseEstimator, RegressorMixin):
    '''
    Implement weighted elastic net
    '''
    
    def __init__(self, l1= 0.5, l2= 0.5, weight =  None,
                 step_size = 1e-3, max_iter = 500, 
                 tol= 1e-3, random_state = 0, fit_intercept = True,
                 method = 'coordinate_descent'):
        '''
        Parameters:
            l1 (float): A float between 0 and 1 for lasso regularization
            l2 (float): A float between 0 and 1 for riage regularization
            weight (array): A 2D weight with the suitable dimensions. Default 
                is identify matrix
            step_size (float): A float that determines the step size
            max_iter (int): The maximun number of iterations
            tol (float): The tolerlance for the solution
            random_state (int): The random state
            fit_intercept (boolean): wthether to calculate the intercept for 
                the model. Default True.
            method (string): The string represents the method in optimization
        Attributes:
            
        Returns: None
        '''
        # assign the parameters
        self.weight = weight
        self.max_iter = max_iter
        self.step_size = step_size
        self.tol = tol
        self.random_state = random_state
        self.l1 = l1
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.method = method
        
        # initize other parameters
        self._coeff = None
        self._n_observations = None
        self._n_features = None
        self._max_peak = None
        self._min_peak = None
        self._gradient = None
        self._cost = None
        self._best_cost = None
        self._best_coeff = None
    
    @staticmethod
    def _get_indicator(length):
        '''
        Args:
            x (array): 
        '''
        indicator = np.zeros(length)
        indicator[np.random.randint(0, length)] = 1
        return indicator
        
    @staticmethod
    def _subgradient_absfunction(x):
        '''
        Args:
            x (array): The array represents the position
            
        Returns: 
            array: a array represents the subderivative of abs function at x
        '''
        sub_derivative = (x != 0)*np.sign(x) # when component != 0
        sub_derivative += (x == 0)*(2*np.random.random() - 1) # when component == 0
        return sub_derivative  

    @staticmethod
    def _soft_thresholding(a, l):
        if a > l:
            return (a - l)
        elif a < -l:
            return a + l
        else:
            return 0
        
        
        
    def _calculate_n_observations(self, X):
        '''
        Args:
            X (array): A matrix that contains data
        
        Returns: None
        '''
        self._n_observations = X.shape[0]
        
    def _calculate_n_features(self, X):
        '''
        Args:
            X (array): A matrix that contains data
            
        Returns: None
        '''
        self._n_features = X.shape[1]
    
    def _calculate_peaks(self, X):
        '''
        Args:
            X (array): A matrix that contains data
            
        Returns: None
        '''
        self._max_peak = np.max(X, axis = 0)
        self._min_peak = np.min(X, axis = 0)
    
    def _initialize_coeff(self, X):
        '''
        Args:
            X (array): A matrix that contians data
            
        Reutrns: None
        '''
        np.random_state = self.random_state
        
        if self._n_features is None:
            self._calculate_n_features(X)
        if (self._min_peak is None) or (self._max_peak is None):
            self._calculate_peaks(X)
        ranges = (self._max_peak - self._min_peak)
        self._coeff = ranges*np.random.random(self._n_features) \
            + self._min_peak
    
    def _initialize_weight(self, X):
        '''
        Args:
            X (array): A matrix that contains data
        
        Returns: None 
        '''
        if self.weight is None:
            if self._n_observations is None:
                self._calculate_n_observations(X)
            self.weight = np.identity(self._n_observations)
            
        
    def _should_stop(self, gradient, iters):
        '''
        Args:
            gradient (array): The current gradient
            iters: The number of iterations
        
        Returns:
            boolean: True for meeting stop conditions. Otherwise returns False
        '''
        if iters > self.max_iter:
            return True
        else:
            # if the gradient fulfils this requirement == min
            return np.linalg.norm(gradient) < \
            np.linalg.norm(self.l1*np.ones(self._n_features)) + self.tol

    def _calculate_k_coeff(self, X, y, k):
        alpha_k = (np.asscalar( self._dual_matrix_y[k,:] \
                              - self._dual_matrix_x[k,:]@self._coeff)\
                  + self._dual_matrix_x[k,k]*self._coeff[k])\
                  /self._dual_matrix_x[k,k]  
        lambda_k = self.l1/self._dual_matrix_x[k,k]/2
        return self._soft_thresholding(alpha_k, lambda_k)
                
    def _calculate_dual_matrix(self, X, y):
        if self._n_features is None:
            self._calculate_n_features(X)
        if self.weight is None:
            self._initialize_weight(X)
        
        I = np.identity(self._n_features)
        self._dual_matrix_x = (np.transpose(X)@self.weight@X + self.l2*I)\
            /(1 + self.l2)
        self._dual_matrix_y = (np.transpose(X)@self.weight@y).reshape(self._n_features, 1)
                
    def _calculate_gradient(self, X, y):
        '''
        Args:
            X (array): A matrix that contians data
            y (array): A array that contains labels
            
        Returns: None
        '''
        from functools import partial
        
        if self._n_observations is None:
            self._calculate_n_observations(X)
        if self._n_features is None:
            self._calculate_n_features(X)
        if self.weight is None:
            self._initialize_weight(X)
        
        def subgradient(x, X, y, W, l1, l2, n_observations, n_features):
            x = x.reshape(n_features, 1)
            y = y.reshape(n_observations, 1)
            return (2*self._dual_matrix_x@x -2*self._dual_matrix_y).flatten()
        
        self._gradient = partial(subgradient, X = X, y = y, W = self.weight,
                                 l1 = self.l1, l2 = self.l2, 
                                 n_observations = self._n_observations, 
                                 n_features = self._n_features)

    def _calculate_cost(self, X, y):
        '''
        Args:
            X (array): A matrix that contians data
            y (array): A array that contains labels
            
        Returns: None
        '''
        from functools import partial
        
        if self._n_observations is None:
            self._calculate_n_observations(X)
        if self._n_features is None:
            self._calculate_n_features(X)
        if self.weight is None:
            self._initialize_weight(X)
        
        def cost(x, X, y, W, l1, l2, n_observations, n_features):
            x = x.reshape(n_features, 1)
            y = y.reshape(n_observations, 1)
            lf1 = np.transpose(x)@self._dual_matrix_x@x
            lf2 = -2*np.transpose(self._dual_matrix_y)@x
            return np.asscalar(lf1 + lf2)+ l1*np.linalg.norm(x, ord=1)
        
        self._cost = partial(cost, X=X, y=y, W =self.weight, 
                             l1 = self.l1, l2=self.l2, 
                             n_observations = self._n_observations, 
                             n_features = self._n_features)
    
    def _calculate_coeff(self, step_size, gradient):
        '''
        Args:
            step_size (float): The step size
            gradient (array): A array 
            
        '''
        self._coeff = self._coeff - step_size*gradient
    
               
    def _estimate_weight(self, X, y):
        '''
        Args:
            X (array): A matrix that contians data
            y (array): A array that contains labels
        '''
        y_predict = self.predict(X)
        y = np.array(y)
        difference = 1/(y - y_predict)**2
        return np.diag(difference.flatten())
    
    def _store_best_results(self, current_cost):
        '''
        Args:
            current_cost (float): The calculated cost
        '''
        if (self._best_cost is None) or (current_cost < self._best_cost):
            # assign the current cost and coeff to the best 
            self._best_cost = current_cost
            self._best_coeff = self._coeff
        
    def _preprocess_data(self, X):
        if self.fit_intercept:
            X = np.hstack( (np.ones((X.shape[0],1)), X ))
        self._calculate_n_features(X)
        return X
        
    def predict(self, X):
        '''
        Args:
            X (array): A matrix that contians data
        '''
        # check is fit had been called
        #check_is_fitted(self, ['X_', 'y_'])
        
        # check X array
        X = check_array(X)
        X = self._preprocess_data(X)
        
        return (np.array(X)@self._best_coeff.reshape(self._n_features,1)).flatten()
    
    def fit(self, X, y):
        '''
        Args:
            X (array): array that contains data
            y (array): array that contains labels
            
        Returns:
            self: object
                
        '''
        
        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        X = self._preprocess_data(X)
        self._calculate_dual_matrix(X,y)
        
        iters = 0
        self._initialize_coeff(X)
        self._calculate_cost(X, y)
        self._calculate_gradient(X, y)
        
        
        if self.method == 'coordinate_descent':
        
            while not self._should_stop(self._gradient(self._coeff), iters):
                iters += 1
                # cyclic coordinate descent
                k = iters%self._n_features
                # update the coeff
                self._coeff[k] = self._calculate_k_coeff(X, y, k)
                # calculate the cost
                current_cost = self._cost(self._coeff)
                # store the current results if it is better than the stored best            
                self._store_best_results(current_cost)       
                print(iters, current_cost, np.linalg.norm(self._gradient(self._coeff)))
            
        elif self.method == 'subgradient_descent':
            
            while not self._should_stop(self._gradient(self._coeff), iters):
                iters += 1
            
                # update the coeff
                self._calculate_coeff(self.step_size, self._gradient(self._coeff))
                current_cost = self._cost(self._coeff)
            
                # store the current results if it is better than the stored best            
                self._store_best_results(current_cost)
                print(iters, self._best_cost, np.linalg.norm(self._gradient(self._coeff)))
        
        # save the total number of iteration
        self._n_iter = iters
        return self
    
if __name__ == "__main__":
    import pandas as pd
    
    df = pd.read_csv('Boston.csv')
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    from sklearn.preprocessing import MinMaxScaler

    # initize a scaler
    scaler_X = MinMaxScaler()

    # fit and transform
    X_scaled = scaler_X.fit_transform(X)
    
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=0)
    
    from sklearn.linear_model import LinearRegression

    # calculate the weight
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_predict_reg = reg.predict(X_train)
    weight = np.diag(1/(y_train - y_predict_reg)**2)
    
    net1 = WEN(tol = 1e-3, step_size = 5e-4, max_iter = 5000, random_state=0,
               fit_intercept=True, method = 'subgradient_descent')
    
    net1.fit(X_train, y_train)
    
    print(net1._best_cost, net1._best_coeff)

    
    
    
        