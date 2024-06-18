from sklearn.linear_model import Ridge, Lasso
import numpy as np 

class ML_model:

    def __init__(self, X_data, y_data, X_ub = None, X_lb = None, typemodel='ridge'):
        """
        Initialize Machine Learning Model with the given data and model type.

        Args:
            X_data (array): Input data.
            y_data (array): Target values.
            X_ub (float, optional): Upper bound. Defaults to None.
            X_lb (float, optional): Lower bound. Defaults to None.
            typemodel (str, optional): Type of the model ('ridge' or 'lasso'). Defaults to 'ridge'.
        """
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
        '''
        Predict output for the given input.
        '''
        return self.model.predict(X)
    
    def update(self, X_new, y_new):
        '''
        Update the model with new data.
        '''
        self._X_data = np.concatenate([self._X_data, X_new], axis=0)
        self._y_data = np.concatenate([self._y_data, y_new], axis=0)

        self.model.fit(self._X_data, self._y_data)
    def nascent_minima_binary(self, X, k=2):
        return np.exp(-k*np.linalg.norm(1-self.predict(X))/self._const)
    
    def nascent_minima(self, X, k=2, error=1):
        '''
        Compute the nascent minima penalization.
        '''
        return np.exp(-k*np.linalg.norm(self.predict(X)*error)/self._const)
    
    def out_count(self, X): 
        '''
        Count the number of points otside the bounds
        ''' 
        if self._X_ub is not None and self._X_lb is not None:
            return np.sum((X > self._X_ub) | (X < self._X_lb))
        elif self._X_ub is not None:
            return np.sum(X > self._X_ub)
        elif self._X_lb is not None:
            return np.sum(X < self._X_lb)
        else: raise ValueError("Upper or lower bound should be provided.")

    def out_pred_ratio(self, X, error=1):  
        '''
        Compute the ratio of predictions outside the bounds.
        '''
        pred  = self.predict(X)*error
        total_points = len(pred)
        out = self.out_count(pred)
        return out / total_points if total_points != 0 else 0.0

    def linear_penality(self, X, error=1):
        '''
        Coompute the linear penality.
        '''
        return 1 - self.out_pred_ratio(X, error) 
    
    def quadratic_penality(self, X, error=1):
        '''
        Compute the quadratic penality.
        '''
        return self.linear_penality(X, error)**2
    
    def exponential_penality(self, X, k=2, error=1):
        '''
        Compute the exponential penality.
        '''
        return np.exp(-k * self.out_pred_ratio(X, error))

    def identity(self, X):
        '''
        Identity function of the prediction 
        '''
        if self.out_pred_ratio(X) > 0: return 0
        else: return 1
    
    def check_inside(self, x):
        """
        Check if prediction of the point is inside the bounds.
        """
        X = self.predict(x)
        if self._X_ub is not None and self._X_lb is not None:
            return not np.any((X > self._X_ub) | (X < self._X_lb))
        elif self._X_ub is not None:
            return not np.any(X > self._X_ub)
        elif self._X_lb is not None:
            return not np.any(X < self._X_lb)
        else:
            raise ValueError("Upper or lower bound should be provided.")

