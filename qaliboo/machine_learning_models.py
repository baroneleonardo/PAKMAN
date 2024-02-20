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

        self._const = np.linalg.norm(self.model.predict(X_data))

    @property
    def X_data(self):
        return self._X_data
    @property
    def y_data(self):
        return self._y_data
    @property
    def typemodel(self):
        return self._type

    def predict(self,X):
        return self.model.predict(X)
    
    def update(self, X_new, y_new):
        self._X_data = np.concatenate([self._X_data, X_new], axis=0)
        self._y_data = np.concatenate([self._y_data, y_new], axis=0)

        self.model.fit(self._X_data, self._y_data)

    def nascent_minima(self, X, k=2):
        #pred = self.predict(X)
        
        return np.exp(-k*np.linalg.norm(self.predict(X))/self._const)
    
    # Ratio of how many points of the q-dimension batch are
    # otside the constraints

    def out_ratio(self, X):  
        if (self._X_ub is not None) and (self._X_lb is not None):
            pred = self.predict(X)
            out_count = np.sum((pred > self._X_ub) | (pred < self._X_lb))
            tot_p = len(pred)
            out_ratio = out_count / tot_p
            return out_ratio
        
        elif self._X_ub is not None:
            pred = self.predict(X)
            out_count = np.sum(pred > self._X_ub)
            tot_p = len(pred)
            out_ratio = out_count / tot_p
            return out_ratio
        
        elif self._X_lb is not None:
            pred = self.predict(X)
            out_count = np.sum(pred < self._X_lb)
            tot_p = len(pred)
            out_ratio = out_count / tot_p
            return out_ratio
        
        else:
            raise ValueError("Upper or lower bound should be provided.")

    # Penality terms
    def linear_penality(self, X):
        return 1 - self.out_ratio(X)
    
    def quadratic_penality(self, X):
        return self.linear_penality(X)**2
    
    def exponential_penality(self, X, k=2):
        return np.exp(-k * self.out_ratio(X))

    def identity(self, X):
        if self.out_ratio(X) > 0: return 0
        else: return 1
    
    def check_inside(self, X):
        if (self._X_ub is not None) and (self._X_lb is not None):
            pred = self.predict(X)
            if (pred > self._X_ub) | (pred < self._X_lb):
                return False
            else: return True
        elif self._X_ub is not None:
            pred = self.predict(X)
            if(pred > self._X_ub):
                return False
            else: return True
        elif self._X_lb is not None:
            if(pred < self._X_lb):
                return False
            else: return True
        else:
            raise ValueError("Upper or lower bound should be provided.")

    def out_count(self, X):  
        if (self._X_ub is not None) and (self._X_lb is not None):
            out_count = np.sum((X > self._X_ub) | (X < self._X_lb))
            return out_count
        elif self._X_ub is not None:
            out_count = np.sum(X > self._X_ub)
            return out_count
        elif self._X_lb is not None:
            out_count = np.sum(X < self._X_lb)
            return out_count
        else: raise ValueError("Upper or lower bound should be provided.")