from sklearn.linear_model import Ridge, Lasso

def train_ml_model(X_data, y_target, typeML = 'ridge'):
    if typeML=='ridge':
        model = Ridge()
    elif typeML=='lasso':
        model== Lasso()
    else:
        raise KeyError('Select a valid Machine Learning Model')
    model.fit(X_data, y_target)

    return model
