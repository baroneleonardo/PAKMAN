import pandas as pd
import numpy as np
from qaliboo import finite_domain
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from examples.abstract_problem import AbstractProblem
import os
from sklearn.model_selection import train_test_split


class XGBoostRegressor(AbstractProblem):
    def __init__(self):
        super().__init__(search_domain = np.array([[3.0, 18.0],   # max_depth (int)
                                                   [50.0, 400.0],     # n_estimators (int)
                                                   [0.01, 1.0],     # reg_lambda (float)
                                                   [0.01, 1.0],    # # colsample_bytree (float)
                                                   [0.01, 1],       # subsample  (float)
                                                   [0.01, 1],       #eta (float)
                                                   [0.01, 10]]),    #min_child_weight(float)     
                                                
                                                         
                         min_value=0.0)
        
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'Features_Variant_1.csv')
        df = pd.read_csv(file_path, header=None)
        df.sample(n=5)
        self.X, self.y = df.loc[:,:52].values, df.loc[:,53].values
        self._init_point = [[6, 50, 1, 1, 1, 0.3, 1]]

    @property
    def init_point(self):
        return self._init_point

    def train(self, x, n_splits=5, objective='reg:squarederror'):
        max_depth, n_estimators, reg_lambda, colsample_bytree, subsample, eta, min_child_weight = x
        maes = []
        r2_is = []
        # Initialize XGBClassifier
        clf = xgb.XGBRegressor( 
            n_estimators=int(n_estimators),
            subsample=subsample,
            max_depth=int(max_depth),
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            eta=eta,
            min_child_weight = min_child_weight,
            early_stopping_rounds=10,
            objective=objective
        )
        # Perform KFold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            evaluation = [(X_train, y_train), (X_test, y_test)]
            clf.fit(X_train, y_train, eval_set=evaluation, verbose=False)
            pred = clf.predict(X_test)
            mae_i = mean_absolute_error(y_test, pred)
            r2_i = r2_score(y_test, pred)
            maes.append(mae_i)
            r2_is.append(r2_i)
        # Calculate mean accuracy and F1 score
        mae = np.mean(maes)
        r2 = np.mean(r2_is) 
        return mae, r2

    


    def evaluate_true(self, x):
        mae, r2 = self.train(x)
        return np.array([mae])
    def evaluate_time(self, x):
        loss, r2 = self.train(x)
        return np.array([r2])

class XGBoostBinary(AbstractProblem):
    def __init__(self):
        super().__init__(search_domain = np.array([[3.0, 18.0],   # max_depth (int)
                                                   [50.0, 400.0],     # n_estimators (int)
                                                   [0.01, 1.0],     # reg_lambda (float)
                                                   [0.01, 1.0],    # # colsample_bytree (float)
                                                   [0.01, 1],       # subsample  (float)
                                                   [0.01, 1],       #eta (float)
                                                   [0.01, 10]]),    #min_child_weight(float)     
                                                
                                                         
                         min_value=0.0)
        
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'blood.csv')
        df = pd.read_csv(file_path, header=0)
        df.drop(columns=['id'], inplace=True)
        self.X  = df.drop('Class', axis=1)
        self.y = df['Class'] - 1
        self._init_point = [[6, 50, 1, 1, 1, 0.3, 1]]

    @property
    def init_point(self):
        return self._init_point

    def train(self, x, n_splits=10, objective='binary:logistic'):
        max_depth, n_estimators, reg_lambda, colsample_bytree, subsample, eta, min_child_weight = x
        accs = []
        f1_is = []
        # Initialize XGBClassifier
        clf = xgb.XGBClassifier( 
            n_estimators=int(n_estimators),
            subsample=subsample,
            max_depth=int(max_depth),
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            eta=eta,
            min_child_weight = min_child_weight,
            early_stopping_rounds=10,
            objective=objective
        )
        # Perform KFold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            evaluation = [(X_train, y_train), (X_test, y_test)]
            clf.fit(X_train, y_train, eval_set=evaluation, verbose=False)
            pred = clf.predict(X_test)
            acc_i = accuracy_score(y_test, pred)
            f1_i = f1_score(y_test, pred)
            accs.append(acc_i)
            f1_is.append(f1_i)
        # Calculate mean accuracy and F1 score
        acc = np.mean(accs)
        f1 = np.mean(f1_is) 
        loss = 1- acc
        return loss, f1

    def evaluate_true(self, x):
        loss, f1 = self.train(x)
        return np.array([loss])
    def evaluate_time(self, x):
        loss, f1 = self.train(x)
        return np.array([f1])
   

class RandomForest(AbstractProblem):

    def __init__(self):
        super().__init__(search_domain = np.array([[2.0, 20.0],  # max_depth (int)
                                                   [2.0,10.0],   # min_samples_split (int)
                                                   [1.0,25.0],]),   # min_samples_leaf (int)
                                                      
                         min_value=0.0)
        
        self.n_estimators = 100
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'data1.csv')
        df = pd.read_csv(file_path)
        self.X = df.drop('Diagnosis')
        self.y = df['Diagnosis']
        #self.y[self.y == 2] = 0
        #self.y[self.y == 1] = 1
    
    def train(self, x):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state = 0)


        max_depth, min_samples_split, min_samples_leaf = x
        
        rf=RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=int(max_depth),
                min_samples_split=int(min_samples_split),
                min_samples_leaf=int(min_samples_leaf))

        
        rf.fit(X_train, y_train)

        pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, pred>0.5)
        loss = 1-accuracy
        return loss
    
    def evaluate_true(self, x):
        loss = self.train(x)
        return np.array([loss])
    
class GradientBoosting(AbstractProblem):

    def __init__(self):
        super().__init__(search_domain = np.array([[0.025, 0.3],       # learning_rate (float)
                                                   [2.0, 25.0],        # max_depth (int)
                                                   [2.0, 20.0],        # min_samples_split
                                                   [0.15, 1.0]]),      # subsample (int)
                                                      
                         min_value=0.0)
        
        self.n_estimators = 100
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'data1.csv')
        df = pd.read_csv(file_path)
        self.X = df.drop('Diagnosis', axis=1)
        self.y = df['Diagnosis']

    
    def train(self, x):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3)


        learning_rate, max_depth, min_samples_split, subsample = x
        
        gb = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=int(learning_rate),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
        )

        
        gb.fit(X_train, y_train)

        pred = gb.predict(X_test)
        accuracy = accuracy_score(y_test, pred>0.5)
        loss = 1-accuracy
        return loss
    
    def evaluate_true(self, x):
        loss = self.train(x)
        return np.array([loss])

