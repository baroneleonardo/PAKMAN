from sklearn.linear_model import Ridge, Lasso
import numpy as np 

class ML_model:
    def __init__(self,
                 X_data,
                 y_data,
                 X_ub = None,
                 typemodel='ridge'):
        
        self._X_data = X_data
        self._y_data = y_data
        self._X_ub = X_ub
        self._type = typemodel

        if self._type=='ridge':
            self.model = Ridge()
        elif self._type=='lasso':
            self.model==Lasso()
        else:
            raise KeyError('Select a valid Machine Learning Model')
        
        self.model.fit(self._X_data, self._y_data)

    @property
    def X_data(self):
        return self._X_data
    @property
    def y_data(self):
        return self._y_data
    

    def predict(self,X):
        return self.model.predict(X)
    
    def update(self, X_new, y_new):
        self._X_data = np.concatenate([self._X_data, X_new], axis=0)
        self._y_data = np.concatenate([self._y_data, y_new], axis=0)

        self.model.fit(self._X_data, self._y_data)

    def nascent_minima(self, X, k=2):
        # TODO of course we can have q>1 points so how we deal with this case?
        return np.exp(-k*self.predict(X))
    
    def identity_ub(self, X):

        if (self._X_ub== None):
            raise KeyError('Upper bound for X is NOT setted')
        
        elif (self.predict(X) < self._X_ub).all():
            return 1
        else:
            return 0

