import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from examples.abstract_problem import AbstractProblem
import os
from sklearn.model_selection import train_test_split
import pickle


class XGBoost(AbstractProblem):
    
    def __init__(self):
        super().__init__(search_domain = np.array([[3.0, 18.0],   # max_depth (int)
                                                   [1.0, 9.0],     # gamma (float)
                                                   [0.0, 1.0],     # reg_lambda (float)
                                                   [0.5, 1.0],   # colsample_bytree (float)
                                                   [0.0, 10.0],    # min_child_weight (int)
                                                   [0.05, 0.3]]), #learning rate (float)  
                         min_value=0.0)
        
        self.default_parameters = np.array([[6, 0, 1, 1, 1, 0.3]])
        self.n_estimators=180
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'data1.csv')
        df = pd.read_csv(file_path)
        self.X = df.drop('Diagnosis', axis=1)
        self.y = df['Diagnosis']
        #self.y[self.y == 2] = 0
        #self.y[self.y == 1] = 1
    
    @property
    def init_params(self):
        return self.default_parameters


    def train(self, x):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3)


        max_depth, gamma, reg_lambda, colsample_bytree, min_child_weight, learning_rate = x

        clf=xgb.XGBClassifier(
                n_estimators = self.n_estimators, 
                max_depth = int(max_depth), 
                gamma = gamma,
                min_child_weight=int(min_child_weight),
                colsample_bytree=colsample_bytree,
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
        self.X = df.drop('Diagnosis', axis=1)
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

class Iris(AbstractProblem):
    def __init__(self):
        super().__init__(search_domain = np.array([[3.0, 18.0],   # max_depth (int)
                                                   [1.0, 9.0],     # gamma (float)
                                                   [0.0, 180.0], # reg_alpha (int)
                                                   [0.0, 1.0],     # reg_lambda (float)
                                                   [0.5, 1.0],   # colsample_bytree (float)
                                                   [0.0, 10.0],    # min_child_weight (int)
                                                   [0.05, 0.3],   #learning rate (float)
                                                   [50, 500]]),      
                         min_value=0.0)
        
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'iris.csv')
        df = pd.read_csv(file_path)
        self.X = df.drop('class', axis=1)
        self.y = df['class']

    def train(self, x):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3)
        max_depth, gamma, reg_alpha, reg_lambda, colsample_bytree, min_child_weight, learning_rate, n_estimators = x

        clf=xgb.XGBClassifier(
                n_estimators = int(n_estimators), 
                max_depth = int(max_depth), 
                gamma = gamma,
                reg_alpha=int(reg_alpha),
                reg_lambda=reg_lambda,
                min_child_weight=int(min_child_weight),
                colsample_bytree=colsample_bytree,
                early_stopping_rounds=10,
                objective='multi::softmax',
                learning_rate=learning_rate,
                subsample=0.8
                )

        evaluation = [( X_train, y_train), ( X_test, y_test)]

        clf.fit(X_train, y_train,
        eval_set=evaluation,
        verbose=False)

        pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        loss = 1-accuracy
        return loss
    

    def evaluate_true(self, x):
        loss = self.train(x)
        return np.array([loss])
    
class IrisRF(AbstractProblem):

    def __init__(self):
        super().__init__(search_domain = np.array([[2.0, 20.0],  # max_depth (int)
                                                   [2.0,10.0],   # min_samples_split (int)
                                                   [1.0,25.0],   # min_samples_leaf (int)
                                                   [50, 500]]),  # n_estimators (int)
                                                      
                         min_value=0.0)
        
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'iris.csv')
        df = pd.read_csv(file_path)
        self.X = df.drop('class', axis=1)
        self.y = df['class']
    
    def train(self, x):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3)


        max_depth, min_samples_split, min_samples_leaf, n_estimators = x
        
        rf=RandomForestClassifier(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                min_samples_split=int(min_samples_split),
                min_samples_leaf=int(min_samples_leaf),
        )

        
        rf.fit(X_train, y_train)

        pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, pred>0.5)
        loss = 1-accuracy
        return loss
    
    def evaluate_true(self, x):
        loss = self.train(x)
        return np.array([loss])

class IrisGB(AbstractProblem):

    def __init__(self):
        super().__init__(search_domain = np.array([[0.025, 0.3],       # learning_rate (float)
                                                   [2.0, 25.0],        # max_depth (int)
                                                   [2.0, 20.0],        # min_samples_split
                                                   [0.15, 1.0],        # subsample (float)
                                                   [50.0, 500.0]]),    # n_estimators (int)  
                                                      
                         min_value=0.0)
        
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'iris.csv')
        df = pd.read_csv(file_path)
        self.X = df.drop('class', axis=1)
        self.y = df['class']
    
    def train(self, x):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3)


        learning_rate, max_depth, min_samples_split, subsample, n_estimators = x
        
        gb = GradientBoostingClassifier(
            n_estimators=int(n_estimators),
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


class CIFRAR10(AbstractProblem):
    
    def __init__(self):
        super().__init__(search_domain = np.array([[3.0, 18.0],   # max_depth (int)
                                                   [1.0, 9.0],     # gamma (float)
                                                   [0.0, 180.0], # reg_alpha (int)
                                                   [0.0, 1.0],     # reg_lambda (float)
                                                   [0.5, 1.0],   # colsample_bytree (float)
                                                   [0.0, 10.0],    # min_child_weight (int)
                                                   [0.05, 0.3],
                                                   [50, 500]]), #learning rate (float)  
                         min_value=0.0)


        current_directory = os.path.dirname(os.path.abspath(__file__))
        test_path = os.path.join(current_directory, 'test_cifar10.pkl')
        train_path = os.path.join(current_directory, 'train_cifar10.pkl')
        with open(train_path, 'rb') as f:
            train_images, train_labels = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_images, test_labels = pickle.load(f)
        self.X_train = train_images / 255.0  # Normalizzazione dei pixel nell'intervallo [0, 1]
        self.X_test = test_images / 255.0
        self.y_train = train_labels.ravel()  # Se le etichette sono ad es. [0, 1, 2, ...], trasformale in un array 1D
        self.y_test = test_labels.ravel()
        self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], -1)

    def train(self, x):
        

        max_depth, gamma, reg_alpha, reg_lambda, colsample_bytree, min_child_weight, learning_rate, n_estimators = x

        clf=xgb.XGBClassifier( 
                max_depth = int(max_depth), 
                gamma = gamma,
                reg_alpha = reg_alpha,
                reg_lambda=reg_lambda,
                min_child_weight=int(min_child_weight),
                colsample_bytree=colsample_bytree,
                early_stopping_rounds=10,
                learning_rate=learning_rate,
                subsample=0.8,
                n_estimators=n_estimators
                )
        
        evaluation = [( self.X_train, self.y_train), ( self.X_test, self.y_test)]

        clf.fit(self.X_train, self.y_train,
        eval_set=evaluation,
        verbose=False)

        pred = clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, pred)
        loss = 1-accuracy
        return loss
    

    def evaluate_true(self, x):
        loss = self.train(x)
        return np.array([loss])


