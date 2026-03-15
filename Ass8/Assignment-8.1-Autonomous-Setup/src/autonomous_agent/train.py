import json
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def main():
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')

    # Default parameters
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 4,
        'random_state': 42
    }

    # Override with hyperparams.json if it exists
    if os.path.exists('hyperparams.json'):
        try:
            with open('hyperparams.json', 'r') as f:
                new_params = json.load(f)
                # Map to standard XGBoost types and filter unknown args
                valid_keys = ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda']
                for k, v in new_params.items():
                    if k in valid_keys:
                        if k in ['n_estimators', 'max_depth']:
                            params[k] = int(v)
                        else:
                            params[k] = float(v)
        except Exception as e:
            print(f'Error reading hyperparams: {e}')

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f'val_rmse: {rmse}')

if __name__ == '__main__':
    main()

