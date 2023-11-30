import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from examples.abstract_problem import AbstractProblem
import os
from sklearn.model_selection import train_test_split
from cifar10_web import cifar10
import pickle


class XGBoost(AbstractProblem):
    
    def __init__(self):
        super().__init__(search_domain = np.array([[3.0, 18.0],   # max_depth (int)
                                                   [1.0, 9.0],     # gamma (float)
                                                   [40.0, 180.0], # reg_alpha (int)
                                                   [0.0, 1.0],     # reg_lambda (float)
                                                   [0.5, 1.0],   # colsample_bytree (float)
                                                   [0.0, 10.0],    # min_child_weight (int)
                                                   [0.05, 0.3]]), #learning rate (float)  
                         min_value=0.0)
        
        self.n_estimators=180
        current_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_directory, 'data1.csv')
        df = pd.read_csv(file_path)
        self.X = df.drop('Diagnosis', axis=1)
        self.y = df['Diagnosis']
        #self.y[self.y == 2] = 0
        #self.y[self.y == 1] = 1

    def train(self, x):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3)


        max_depth, gamma, reg_alpha, reg_lambda, colsample_bytree, min_child_weight, learning_rate = x

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
    
'''
class CIFRAR10(AbstractProblem):
    
    def __init__(self):
        super().__init__(search_domain = np.array([[3.0, 18.0],   # max_depth (int)
                                                   [1.0, 9.0],     # gamma (float)
                                                   [40.0, 180.0], # reg_alpha (int)
                                                   [0.0, 1.0],     # reg_lambda (float)
                                                   [0.5, 1.0],   # colsample_bytree (float)
                                                   [0.0, 10.0],    # min_child_weight (int)
                                                   [0.05, 0.3]]), #learning rate (float)  
                         min_value=0.0)
        
        self.n_estimators=10

        data_dir = '/home/lbarone/QALIBOO/examples/RealProblem/cifrar10'

        X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)
        
        RESHAPED = 3072
        self.X_train = X_train.reshape(50000, RESHAPED)
        self.X_test = X_test.reshape(10000, RESHAPED)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.y_train = y_train.flatten()
        self.y_test = y_test.flatten()

        # Normalize the dataset:
        self.X_train /= 255.
        self.X_test /= 255.
        

    def train(self, x):
        

        max_depth, gamma, reg_alpha, reg_lambda, colsample_bytree, min_child_weight, learning_rate = x

        clf=xgb.XGBClassifier(
                n_estimators = self.n_estimators, 
                max_depth = int(max_depth), 
                gamma = gamma,
                reg_alpha = int(reg_alpha),
                reg_lambda=reg_lambda,
                min_child_weight=int(min_child_weight),
                colsample_bytree=int(colsample_bytree),
                early_stopping_rounds=10,
                objective='multi::softmax',
                learning_rate=learning_rate,
                subsample=0.8
                )

        clf.fit(self.X_train, self.y_train, verbose=False)

        pred = clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, pred)
        loss = 1-accuracy
        return loss
    

    def evaluate_true(self, x):
        loss = self.train(x)
        return np.array([loss])


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(data_dir):
    data_batches = []
    labels_batches = []

    # Load data batches
    for i in range(1, 6):
        batch = unpickle(f'{data_dir}/data_batch_{i}')
        data_batches.append(batch[b'data'])
        labels_batches.append(batch[b'labels'])

    test_batch = unpickle(f'{data_dir}/test_batch')
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    train_data = np.concatenate(data_batches)
    train_data = train_data.reshape((len(train_data), 3, 32, 32)).transpose(0, 2, 3, 1)
    train_labels = np.concatenate(labels_batches)

    return train_data, train_labels, test_data, test_labels

# Specify the path to your CIFAR-10 dataset directory

'''
