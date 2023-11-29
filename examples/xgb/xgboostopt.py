import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from examples.abstract_problem import AbstractProblem
import os
from sklearn.model_selection import train_test_split


class XGBoost(AbstractProblem):
    
    def __init__(self):
        super().__init__(search_domain = np.array([[3, 18],   # max_depth (int)
                                                   [1,9],     # gamma (float)
                                                   [40, 180], # reg_alpha (int)
                                                   [0,1],     # reg_lambda (float)
                                                   [0.5,1],   # colsample_bytree (float)
                                                   [0,10],    # min_child_weight (int)
                                                   [0.05, 0.3]]), #learning rate (float)  
                         min_value=0.0)
        
        self.n_estimators=180
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'data.csv')
        df = pd.read_csv(file_path)
        self.X = df.drop('Channel', axis=1)
        self.y = df['Channel']
        self.y[self.y == 2] = 0
        self.y[self.y == 1] = 1

    def train(self, x):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state = 0)


        max_depth, gamma, reg_alpha, reg_lambda, colsample_bytree, min_child_weight, learning_rate = x
        reg_alpha = int(reg_alpha)
        min_child_weight = int(min_child_weight)

        clf=xgb.XGBClassifier(
                n_estimators = self.n_estimators, 
                max_depth = int(max_depth), 
                gamma = gamma,
                reg_alpha = int(reg_alpha),
                reg_lambda=reg_lambda,
                min_child_weight=int(min_child_weight),
                colsample_bytree=int(colsample_bytree),
                early_stopping_rounds=10,
                eval_metric="auc",
                learning_rate=learning_rate,
                )

        evaluation = [( X_train, y_train), ( X_test, y_test)]

        clf.fit(X_train, y_train,
        eval_set=evaluation,
        verbose=False)

        pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, pred>0.5)
        loss = 1-accuracy
        return loss
    

    def evaluate_true(self, x):
        loss = self.train(x)
        return np.array([loss])
