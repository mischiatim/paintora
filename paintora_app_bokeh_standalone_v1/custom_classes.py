import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ModelTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, model):
        self.predictor = model
            
    def fit(self, X, y):
        # Fit the stored predictor.
        self.predictor.fit(X, y)
        return self
    
    def transform(self, X):
        # Use predict on the stored predictor as a "transformation".
        # reshape(-1,1) is required to return a 2-D array which is expected by scikit-learn.
        return np.array(self.predictor.predict(X)).reshape(-1,1)
    

def stringlist_to_dict(stringlist):
        dict = {}
        for string in stringlist:
            dict[string]=1
        return dict
    
class DictEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X will be a pandas series of strings representing lists of strings. Return a pandas series of dictionaries
        return X.apply(eval).apply(stringlist_to_dict)
    

class TagsEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X will be a pandas series of strings representing lists of strings. Return a pandas series of single strings containing all individual strings joined by a space
        return X.apply(eval).apply(lambda stringlist:' '.join(stringlist))