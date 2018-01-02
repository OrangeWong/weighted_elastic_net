# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:59:23 2018

@author: richa
"""
import numpy as np

class WEN():
    '''
    Implement weighted elastic net
    '''
    
    def __init__(self, l1= 0.5, l2= 0.5, weight =  None,
                 step_size = 1e-3, max_iter = 50, 
                 tol= 1e-4, random_state = 0):
        '''
        Args:
            l1 (float): A float between 0 and 1 for lasso regularization
            l2 (float): A float between 0 and 1 for riage regularization
            weight (array): A 2D weight with the suitable dimensions. Default 
            is identify matrix
            step_size (float): A float that determines the step size
            max_iter (int): The maximun number of iterations
            tol (float): The tolerlance for the solution
            random_state (int): The random state
            
        Returns: None
        '''
        self._weight = weight
        self._max_iter = max_iter
        self._step_size = step_size
        self._tol = tol
        self._random_state = random_state
        self._coeff = None
        self._num_observations = None
        self._num_features = None
        self._max_peak = None
        self._min_peak = None
        self._l1 = l1
        self._l2 = l2
        self._gradient = None
        self._cost = None
        self._response = None
                
    def _calculate_num_observations(self, X):
        '''
        Args:
            X (array): A matrix that contains data
        
        Returns: None
        '''
        self._num_observations = (np.array(X)).shape[0]
        
    def _calculate_num_features(self, X):
        '''
        Args:
            X (array): A matrix that contains data
            
        Returns: None
        '''
        self._num_features = (np.array(X)).shape[1]
    
    def _calculate_peaks(self, X):
        '''
        Args:
            X (array): A matrix that contains data
            
        Returns: None
        '''
        self._max_peak = np.max(np.array(X), axis = 0)
        self._min_peak = np.min(np.array(X), axis = 0)
    
    def _initialize_coeff(self, X):
        '''
        Args:
            X (array): A matrix that contians data
            
        Reutrns: None
        '''
        np.random_state = self._random_state
        
        if self._num_features is None:
            self._calculate_num_features(X)
        if (self._min_peak is None) or (self._max_peak is None):
            self._calculate_peaks(X)
        ranges = (self._max_peak - self._min_peak).reshape(self._num_features,1)
        self._coeff = ranges*np.random.random(size = (self._num_features, 1)) \
            + self._min_peak.reshape(self._num_features, 1)
    
    def _initialize_weight(self, X):
        '''
        Args:
            X (array): A matrix that contains data
        
        Returns: None 
        '''
        if self._num_observations is None:
            self._calculate_num_observations(X)
        if self._weight is None:
            self._weight = np.identity(self._num_observations)
            
        
    def _should_stop(self, old_coeff, iters):
        '''
        Args:
            old_coeff (array): The old coefficients 
            iters: The number of iterations
        
        Returns:
            boolean: True for meeting stop conditions. Otherwise returns False
        '''
        if iters > self._max_iter:
            return True
        else:
            if old_coeff is None:
                return False
            else:
                return (np.abs(old_coeff - self._coeff).sum() < self._tol)
            
    def _OLS_coeff(self, X, y):
        '''
        Args:
            X (array): A matrix that contians data
            y (array): A array that contains labels
            
        Returns: 
            array: a array with the same size of self._coeff
        '''
        y = np.array(y).reshape(self._num_observations, 1)
        X = np.array(X)
        return np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y
        
    def _calculate_gradient(self, X, y):
        '''
        Args:
            X (array): A matrix that contians data
            y (array): A array that contains labels
            
        Returns: 
            array: a array with the same size of self._coeff
        '''
        from functools import partial
        
        if self._num_observations is None:
            self._calculate_num_observations(X)
        if self._num_features is None:
            self._calculate_num_features(X)
        sub_derivative = self._subgradient_absfunction(self._coeff)
        
        def subgradient(x, X, y, W, l1, l2, num_observations, num_features,
                        sub_derivative):
            y = np.array(y).reshape(num_observations, 1)
            X = np.array(X)
            I = np.identity(self._num_features)
        
            g1 = 2*(np.transpose(X)@W@X + l2*I)@x/(1 + l2)
            g2 = -2*np.transpose(X)@W@y
            g3 = l1*sub_derivative
            return g1 + g2 + g3
        
        self._gradient = partial(subgradient, X = X, y = y, W = self._weight,
                                 l1 = self._l1, l2 = self._l2, 
                                 num_observations = self._num_observations, 
                                 num_features = self._num_features,
                                 sub_derivative = sub_derivative)

    def _calculate_cost(self, X, y):
        '''
        Args:
            X (array): A matrix that contians data
            y (array): A array that contains labels
            
        Returns: 
            float : a float represents the loss
        '''
        from functools import partial
        
        if self._num_observations is None:
            self._calculate_num_observations(X)
        if self._num_features is None:
            self._calculate_num_features(X)
        if self._weight is None:
            self._initialize_weight(X)
        if self._weight is None:
            self._initialize_weight(X)
        
        def cost(x, X, y, W, l1, l2, num_observations, num_features):
            y = np.array(y).reshape(num_observations, 1)
            X = np.array(X)
            I = np.identity(num_features)
            lf1 = np.transpose(x)@(np.transpose(X)@W@X + l2*I)/(1+ l2)@x
            lf2 = -2*np.transpose(y)@W@X@x 
            lf3 = l1*np.linalg.norm(x, ord=1)
            return np.asscalar(lf1 + lf2)+ lf3
        
        self._cost = partial(cost, X=X, y=y, W =self._weight, 
                             l1 = self._l1, l2=self._l2, 
                             num_observations = self._num_observations, 
                             num_features = self._num_features)
    
    def _calculate_coeff(self, X, y, gradient):
        '''
        Args:
            X (array): A matrix that contians data
            y (array): A array that contains labels
            
        '''
        self._coeff = self._coeff - self._step_size*gradient
    
    def _calculate_response(self, X):
        '''
        Args:
            X (array): A matrix that contians data
        '''
        self._response = np.array(X)@self._coeff
        
        
    def _calculate_weight(self, X, y):
        '''
        Args:
            X (array): A matrix that contians data
            y (array): A array that contains labels
        '''
        self._calculate_response(X)
        y = np.array(y).reshape(self._num_observations, 1)
        difference = 1/(y - self._response)**2
        return np.diag(difference.flatten())
    
    @staticmethod
    def _get_indicator(x):
        a = np.zeros_like(x)
        a[np.random.randint(0, x.shape[0]),0]= 1
        return a
        
        
    @staticmethod
    def _subgradient_absfunction(x):
        '''
        Args: None
            
        Returns: 
            array: a array with the same size of self._coeff
        '''
        sub_derivative = (x != 0)*np.sign(x) # when component != 0
        sub_derivative += (x == 0)*(2*np.random.random() - 1) # when component == 0
        return sub_derivative  
    
    def fit(self, X, y):
        '''
        Args:
            X (array): array that contains data
            y (array): array that contains labels
    
        '''
        iters = 0
        best_coeff = None
        best_cost = None
        old_coeff = None
        old_cost = None
                
        self._initialize_coeff(X)
        self._calculate_cost(X, y)
        self._calculate_gradient(X, y)
        
        while not self._should_stop(old_coeff, iters):
            iters += 1
            # assign the old 
            old_coeff = self._coeff
            old_cost = self._cost(self._coeff)
            
            self._calculate_coeff(X, y, self._gradient(self._coeff))
            
            if best_cost is None:
                best_cost = self._cost(self._coeff)
                best_coeff = self._coeff
            else:
                if self._cost(self._coeff) < best_cost:
                    best_cost = self._cost(self._coeff)
                    best_coeff = self._coeff
             
            print(iters, old_cost, best_cost, np.max(self._weight))
    
if __name__ == "__main__":
    import numpy as np
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
    
    net1 = WEN(step_size = 1e-3, max_iter = 5000, random_state=0)
    net1.fit(X_train, y_train)