import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import statistics as stats

class KNNClassifier(BaseEstimator):
    
    # __init__ accepts in input the value of the number of the neighbors
    # This value is assigned to the homonymous parameter of the class
    def __init__(self, k: int = 3) -> None:
        self.k = k

    # fit() accepts the training set and the relative labels
    # If necessary, the labels are converted to a numpy array
    # Both input paramaters are assigned to the homonymous parameters of the class
    def fit(self, X_train:pd.DataFrame, y_train:pd.DataFrame | np.ndarray) -> object:
        
        self.X_trainset = X_train.to_numpy()
        if isinstance(y_train, pd.DataFrame):
            self.y_trainset = y_train.to_numpy()
        else:
            self.y_trainset = y_train
        return self
    
    # predict() accepts the test set as input parameter
    # For each element in the test set
        # The distances with all the elements in the training set are computed
        # The labels of the k nearest elements are retrievd
        # The most common label is calculated and appended to the returned array
    def predict(self, X_test:pd.DataFrame) -> pd.Series:
        
        indexes = X_test.index.to_list()
        X_test = X_test.to_numpy()
        final_prediction = []
        
        for test_sample in X_test:
            distances = np.linalg.norm(self.X_trainset - test_sample, axis = 1)
            k_labels = self.y_trainset[np.argsort(distances)][:self.k]
            most_common_label = stats.mode(k_labels.flatten())
            final_prediction.append(most_common_label)    
            
        return pd.Series(data = final_prediction, index = indexes)      