import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import xgboost as xgb

from examples.abstract_problem import AbstractProblem

seed = 42

class RF(AbstractProblem):
    def __init__(self):
        super().__init__(search_domain = np.array([[2.0, 20.0],  # max_depth (int)
                                                    [2.0,10.0],   # min_samples_split (int)
                                                    [1.0,25.0],  # min_samples_leaf (int)
                                                    [50, 500]]), # n_estimators(int)   
                                                        
                            min_value=0.0)
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'data_dausare.csv')
        data = pd.read_csv(file_path)
        data = data.set_index('Datetime')
        data = data.dropna(subset = ["NO2(GT)"])
        data = data.dropna(subset = ["T"])
        self.x = data.drop(['NO2(GT)'], axis = 1)
        self.y = pd.DataFrame(data['NO2(GT)'])
        # self.init_point = [] # Metto il punto base dell'algoritmo da li parto e confronto
    def train(self, x):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.3, random_state = seed, shuffle = True)
        
        max_depth, min_samples_split, min_samples_leaf, n_estimators = x

        clf=RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf))
        
        scaler = StandardScaler()
        scaler_y = StandardScaler()
        clf.fit(scaler.fit_transform(x_train), np.ravel(scaler_y.fit_transform(y_train)))
        y_hat = clf.predict(scaler.transform(x_test))
        rmse = metrics.mean_squared_error(np.ravel(scaler_y.transform(y_test)), y_hat, squared=False)
        return rmse

    def evaluate_true(self, x):
        rmse = self.train(x)
        return np.array([rmse])
    
class XGB(AbstractProblem):
    def __init__(self):
        super().__init__(search_domain = np.array([[3.0, 18.0],   # max_depth (int)
                                                   [1.0, 9.0],    # gamma (float)
                                                   [0.0, 1.0],    # reg_lambda (float)
                                                   [0.5, 1.0],    # colsample_bytree (float)
                                                   [0.0, 10.0],   # min_child_weight (int)
                                                   [0.05, 0.3],   #learning rate (float)
                                                   [50, 500]]),   # n_estimators (int)   
                         min_value=0.0)
        
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'data_dausare.csv')
        data = pd.read_csv(file_path)
        data = data.set_index('Datetime')
        data = data.dropna(subset = ["NO2(GT)"])
        data = data.dropna(subset = ["T"])
        self.x = data.drop(['NO2(GT)'], axis = 1)
        self.y = pd.DataFrame(data['NO2(GT)'])

    def train(self, x):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.3, random_state = seed, shuffle = True)
        
        max_depth, gamma, reg_lambda, colsample_bytree, min_child_weight, learning_rate, n_estimators = x

        clf=xgb.XGBRegressor(
                n_estimators = int(n_estimators), 
                max_depth = int(max_depth), 
                gamma = gamma,
                reg_lambda=reg_lambda,
                min_child_weight=int(min_child_weight),
                colsample_bytree=colsample_bytree,
                eval_metric="rmse",
                learning_rate=learning_rate,
                )

        
        scaler = StandardScaler()
        scaler_y = StandardScaler()
        clf.fit(scaler.fit_transform(x_train), np.ravel(scaler_y.fit_transform(y_train)))
        y_hat = clf.predict(scaler.transform(x_test))
        rmse = metrics.mean_squared_error(np.ravel(scaler_y.transform(y_test)), y_hat, squared=False)
        return rmse

    def evaluate_true(self, x):
        rmse = self.train(x)
        return np.array([rmse])
      