from sklearn.linear_model import Ridge, Lasso
import numpy as np 

class ML_model:
    def __init__(self,
                 X_data,
                 y_data,
                 X_ub = None,
                 X_lb = None,
                 typemodel='ridge'):
        
        self._X_data = X_data
        self._y_data = y_data
        self._X_ub = X_ub
        self._type = typemodel
        self._X_lb = X_lb

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
        return np.exp(-k*np.linalg.norm(self.predict(X)))
    
    def identity(self, X):
        if (self._X_ub == None and self._X_lb == None):
            raise KeyError('No constraints defined!')
        pred = self.predict(X)
        if (self._X_lb == None):
            if (pred < self._X_ub).all():
                return 1
            else: 
                return 0
        elif (self._X_ub == None):
            if (self._X_lb < pred).all():
                return 1
            else:
                return 0
        else:
            if (pred < self._X_ub).all() and (self._X_lb < pred).all():
                return 1
            else:
                return 0
