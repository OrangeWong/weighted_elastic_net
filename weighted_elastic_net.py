# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:59:23 2018

@author: richa
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class WEN(BaseEstimator, RegressorMixin):
    '''
    Implement weighted elastic net
    '''
    
    def __init__(self, l1= 0.5, l2= 0.5, weight =  None,
                 step_size = 1e-3, max_iter = 50, 
                 tol= 1e-3, random_state = 0):
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
        # assign the parameters
        self.weight = weight
        self.max_iter = max_iter
        self.step_size = step_size
        self.tol = tol
        self.random_state = random_state
        self.l1 = l1
        self.l2 = l2
        
        # initize other parameters
        self._coeff = None
        self._num_observations = None
        self._num_features = None
        self._max_peak = None
        self._min_peak = None
        self._gradient = None
        self._cost = None
        self._best_cost = None
        self._best_coeff = None
    
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
        np.random_state = self.random_state
        
        if self._num_features is None:
            self._calculate_num_features(X)
        if (self._min_peak is None) or (self._max_peak is None):
            self._calculate_peaks(X)
        ranges = (self._max_peak - self._min_peak)
        self._coeff = ranges*np.random.random(self._num_features) \
            + self._min_peak
    
    def _initialize_weight(self, X):
        '''
        Args:
            X (array): A matrix that contains data
        
        Returns: None 
        '''
        if self._num_observations is None:
            self._calculate_num_observations(X)
        if self.weight is None:
            self.weight = np.identity(self._num_observations)
            
        
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
            return (np.linalg.norm(gradient)/self._num_features) < self.tol
        
    def _calculate_gradient(self, X, y):
        '''
        Args:
            X (array): A matrix that contians data
            y (array): A array that contains labels
            
        Returns: None
        '''
        from functools import partial
        
        if self._num_observations is None:
            self._calculate_num_observations(X)
        if self._num_features is None:
            self._calculate_num_features(X)
        if self.weight is None:
            self._initialize_weight(X)
        
        sub_derivative = self._subgradient_absfunction(self._coeff)
        
        def subgradient(x, X, y, W, l1, l2, num_observations, num_features,
                        sub_derivative):
            
            x = x.reshape(num_features, 1)
            y = np.array(y).reshape(num_observations, 1)
            X = np.array(X)
            I = np.identity(self._num_features)
            
            g1 = 2*(np.transpose(X)@W@X + l2*I)@x/(1 + l2)
            g2 = -2*np.transpose(X)@W@y
            g3 = l1*sub_derivative.reshape(num_features, 1)
            
            return (g1 + g2 + g3).flatten()
        
        self._gradient = partial(subgradient, X = X, y = y, W = self.weight,
                                 l1 = self.l1, l2 = self.l2, 
                                 num_observations = self._num_observations, 
                                 num_features = self._num_features,
                                 sub_derivative = sub_derivative)

    def _calculate_cost(self, X, y):
        '''
        Args:
            X (array): A matrix that contians data
            y (array): A array that contains labels
            
        Returns: None
        '''
        from functools import partial
        
        if self._num_observations is None:
            self._calculate_num_observations(X)
        if self._num_features is None:
            self._calculate_num_features(X)
        if self.weight is None:
            self._initialize_weight(X)
        
        def cost(x, X, y, W, l1, l2, num_observations, num_features):
            x = x.reshape(num_features, 1)
            y = np.array(y).reshape(num_observations, 1)
            X = np.array(X)
            I = np.identity(num_features)
            lf1 = np.transpose(x)@(np.transpose(X)@W@X + l2*I)/(1+ l2)@x
            lf2 = -2*np.transpose(y)@W@X@x 
            lf3 = l1*np.linalg.norm(x, ord=1)
            return np.asscalar(lf1 + lf2)+ lf3
        
        self._cost = partial(cost, X=X, y=y, W =self.weight, 
                             l1 = self.l1, l2=self.l2, 
                             num_observations = self._num_observations, 
                             num_features = self._num_features)
    
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
        
    @staticmethod
    def _get_indicator(x):
        a = np.zeros_like(x)
        a[np.random.randint(0, x.shape[0])]= 1
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
    
    def predict(self, X):
        '''
        Args:
            X (array): A matrix that contians data
        '''
        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        
        # check X array
        X = check_array(X)
        
        return (np.array(X)@self._coeff.reshape(self._num_features,1)).flatten()
    
        
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
        self.X_ = X
        self.y_ = y
        
        iters = 0
        self._initialize_coeff(self.X_)
        self._calculate_cost(self.X_, self.y_)
        self._calculate_gradient(self.X_, self.y_)
        
        while not self._should_stop(self._gradient(self._coeff), iters):
            iters += 1
            
            # update the coeff
            self._calculate_coeff(self.step_size, self._gradient(self._coeff))
            current_cost = self._cost(self._coeff)
            # store the current results if it is better than the stored best            
            self._store_best_results(current_cost)
            #print(iters, self._best_cost, np.linalg.norm(self._gradient(self._coeff))/self._num_features)
        
        return self
    
    def fit_CD(self, X, y):
        import scipy as sp
        from functools import partial
                
        iters = 0
        self._initialize_coeff(X)
        self._calculate_cost(X, y)
        self._calculate_gradient(X, y)
        
        while not self._should_stop(self._gradient(self._coeff), iters):
            iters += 1
            # determine the position and direction for line search
            xk = self._coeff.flatten()
            pk = self._get_indicator(self._coeff)*self._gradient(self._coeff)
            # define a line search function to minimize
            def cost(alpha, xk, pk):
                return self._cost(xk-alpha*pk)
            # minimization
            res = sp.optimize.minimize_scalar(partial(cost, xk = xk, pk=pk))
            # update the coeff
            self._calculate_coeff(res.x, pk)
            current_cost = self._cost(self._coeff)
            # store the best 
            self._store_best_results(current_cost)
            #print(iters, res.x, self._best_cost, np.linalg.norm(self._gradient(self._coeff))/self._num_features)
            
            
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
    
    #net1 = WEN(step_size = 1e-3, max_iter = 5000, random_state=0)
    #net1.fit(X_train, y_train)
    
    #print(net1.predict(X_train) - y_train)
    
    from sklearn.model_selection import GridSearchCV
    
    tuned_params = {'l1': [.2,.4,.6, .8, 1], 'l2': [.2, .4, .6, .8, 1]}
    
    gs = GridSearchCV(WEN(), tuned_params)
    
    gs.fit(X_train, y_train)
    
    gs.best_params_
    
    #from sklearn.utils.estimator_checks import check_estimator
    
    #check_estimator(WEN)
        