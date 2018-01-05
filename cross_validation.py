# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:00:27 2018

@author: richa
"""
import numpy as np

class K_Fold():
    '''
    K-folds cross validation
    '''
    def __init__(self, n_splits, shuffle=True, random_state = 0):
        '''
        Args:
            n_splits (int): the number of splits for k-folds
            shuffle (Boolean): whether to shuffle the data
            random_state (int): the random seed
        '''
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        '''
        '''
        self._n_samples = X.shape[0]
        self._indics = np.arange(self._n_samples)
        if self.shuffle:
            permutation = np.random.permutation(self._n_samples)
            self._indics = self._indics[permutation]
        self._fold_sizes = (self._n_samples//self.n_splits)*\
            np.ones(self.n_splits, dtype = np.int)
        # relocate the remaining data
        self._fold_sizes[:self._n_samples%self.n_splits] += 1
        pointer = 0
        for fold_size in self._fold_sizes:
            start, stop = pointer, pointer+fold_size
            mask = np.zeros(self._n_samples, dtype = bool)
            mask[np.array(range(start, stop))] = True
            test_index = self._indics[mask]
            train_index = self._indics[~mask]
            yield train_index, test_index
            pointer = stop
            
            
    def get_n_splits(self):
        return self.n_splits
    
        
class Bootstrap():

	def __init__(self, df, size, N):
		self.df = df
		self.size = size
		self.N = N

	def get_bootstraps(self):
		"""
		Get bootstrap samples in df format, size = size of
		each bootstrap sample, N = number of bootstrap samples
		"""
		bootstrapped_dfs = []
		for i in range(self.N):
			# randomly select rows from original DF until size is met
			temp = pd.DataFrame()
			for j in range(self.size):
				# uniformly sample with replacement from dataset (there will be duplicate rows)
				rand_index = np.random.randint(0, len(self.df))
				# append row to new bootstrap sample dataframe
				temp = temp.append(self.df.iloc[rand_index, :])[self.df.columns.tolist()]

			# this list is a collection of the bootstrap sample dfs
			bootstrapped_dfs.append(temp)

		return bootstrapped_dfs
    
    
if __name__ == "__main__":
    kfold = K_Fold(n_splits=3)
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
    
    
    for test, train in kfold.split(X_train):
        print(test, len(train))
    
    print(X_train.shape)
    