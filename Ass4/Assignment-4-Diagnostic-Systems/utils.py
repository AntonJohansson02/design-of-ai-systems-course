import pickle
from sklearn.model_selection import train_test_split


def LoadData():
    with open ('wdbc.pkl', 'rb') as f:
        data = pickle.load(f)

  
    X = data.drop('malignant', axis=1)
    y = data['malignant']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    return X_train, X_test, y_train, y_test

LoadData()