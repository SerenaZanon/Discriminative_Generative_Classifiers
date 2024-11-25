import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from scipy.stats import beta

class NaiveBayesClassifier(BaseEstimator):
    
    # __init__ calls the super constructor
    def __init__(self) -> None:
        super().__init__()

    # fit() accepts the training set and the relative labels
    # Both input paramaters are assigned to the homonymous parameters of the class
    # For each label the alpha and the beta paramters are computed and the frequency is calculated
    def fit(self, X_train:pd.DataFrame, y_train: np.ndarray) -> None:
        
        self.X_trainset = X_train
        self.y_trainset = y_train
        
        self.class_parameters={}
        
        labels = list(set(self.y_trainset))
        
        for label in labels:
            
            images = self.X_trainset[self.y_trainset == label]
            
            label_mean = images.mean(axis = 0)
            label_variance = images.var(axis = 0)
            
            param_k = ((label_mean) * (1 - label_mean) / label_variance) - 1
            param_alpha = param_k * label_mean
            param_beta = param_k * (1 - label_mean)
            
            min_alpha = min(param_alpha[param_alpha > 0])
            param_alpha[param_alpha <= 0] = min_alpha
            
            min_beta = min(param_beta[param_beta > 0])
            param_beta[param_beta <= 0] = min_beta
            
            total_elements = self.y_trainset.size
            label_elements = images.size
            
            label_frequency = label_elements / total_elements
            
            self.class_parameters[label] = {'alpha': param_alpha, 
                                            'beta': param_beta,
                                            'frequency': label_frequency}
            

    # predict() accepts the test set as input parameter
    # For each element on the test sample
        # For each possible label
            # The probability of the label is computed and if it is the maximum obtained until now it is saved
        # The label with the major probability is appended to the returned array
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
       
        distorsion = 0.1
        final_prediction = []
        index = []
       
        for row, test_sample in X_test.iterrows():

            max_prob = 0
            max_class = -1
            test_sample = test_sample.to_numpy()
             
            for n in range(10):

                class_parameters = self.class_parameters[n]
                
                param_alpha = class_parameters['alpha']
                param_beta = class_parameters['beta']
                parameter_freq = class_parameters['frequency']
                
                beta_probabilities = beta.cdf(test_sample + distorsion, param_alpha, param_beta) - beta.cdf(test_sample - distorsion, param_alpha, param_beta)              
                np.nan_to_num(beta_probabilities, copy = False, nan = 1.0)
                probability = parameter_freq * np.prod(beta_probabilities)
                
                if probability > max_prob:
                    max_prob = probability
                    max_class = n
            
            final_prediction.append(max_class)
            index.append(row)
            
        return pd.Series(data = final_prediction, index = index)