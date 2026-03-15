import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "prepared"
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

model = XGBRegressor(
    n_estimators=1400,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
val_pred = model.predict(X_val)
rmse = root_mean_squared_error(y_val, val_pred)
print(f"val_rmse: {rmse:.5f}")
