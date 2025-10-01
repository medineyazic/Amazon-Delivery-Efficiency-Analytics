import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("amazon_imputed.csv")

y = df["Delivery_Time"]

X = df.drop(columns=["Delivery_Time", "Delivery_Category", "Delivery_Cluster"])

X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=412
)

cv = KFold(n_splits=5, shuffle=True, random_state=412)
print(f"Train set shape (X): {X_train.shape}")
print(f"Test set shape  (X): {X_test.shape}")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
lr_model = LinearRegression()

# 5-fold CV RMSE
cv_rmse = np.sqrt(-cross_val_score(
    lr_model, X_train, y_train,
    scoring="neg_mean_squared_error", cv=cv
))

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Cross-validated RMSE: {cv_rmse.mean():.4f}")
print(f"Test RMSE          : {rmse:.4f}")
print(f"Test MAE           : {mae:.4f}")
print(f"Test R²            : {r2:.4f}")
print(f"Test MAPE: {mape:.2f}%")

coef_df = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': lr_model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(coef_df.head(10))
residuals = y_test - y_pred

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=50, color="slateblue", alpha=0.7)
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution (Linear Regression)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.3, color="gray")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Delivery Time")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.show()

#ANN
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# MAPE fonksiyonu (bir defa tanımlaman yeterli)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ANN pipeline – optimize edilmiş
ann_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ann", MLPRegressor(
        hidden_layer_sizes=(64, 32),        
        activation='relu',
        solver='adam',
        max_iter=300,                      
        early_stopping=True,             
        random_state=412
    ))
])


cv_rmse_ann = np.sqrt(-cross_val_score(
    ann_pipeline, X_train, y_train,
    scoring="neg_mean_squared_error", cv=5
))
print(f"CV RMSE (ANN): {cv_rmse_ann.mean():.4f}")

ann_pipeline.fit(X_train, y_train)
y_pred_ann = ann_pipeline.predict(X_test)

rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))
mae_ann = mean_absolute_error(y_test, y_pred_ann)
r2_ann = r2_score(y_test, y_pred_ann)
mape_ann = mean_absolute_percentage_error(y_test, y_pred_ann)

print(f"Test RMSE (ANN): {rmse_ann:.4f}")
print(f"Test MAE  (ANN): {mae_ann:.4f}")
print(f"Test R²   (ANN): {r2_ann:.4f}")
print(f"Test MAPE (ANN): {mape_ann:.2f}%")

import matplotlib.pyplot as plt
import seaborn as sns


residuals_ann = y_test - y_pred_ann


plt.figure(figsize=(7, 6))
sns.histplot(residuals_ann, bins=40, kde=True, color="skyblue")
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residuals Distribution of ANN Model")
plt.grid(True)
plt.tight_layout()
plt.show()

#SVM
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# SVM pipeline
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVR(kernel="rbf", C=10, epsilon=1.0))
])

# Cross-validated RMSE
cv_rmse_svm = np.sqrt(-cross_val_score(
    svm_pipeline, X_train, y_train,
    scoring="neg_mean_squared_error", cv=cv
))
print(f"CV RMSE (SVM): {cv_rmse_svm.mean():.4f}")

# Train the model
svm_pipeline.fit(X_train, y_train)
y_pred_svm = svm_pipeline.predict(X_test)

# Performance metrics
rmse_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm))
mae_svm = mean_absolute_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)
mape_svm = mean_absolute_percentage_error(y_test, y_pred_svm)

print(f"Test RMSE (SVM): {rmse_svm:.4f}")
print(f"Test MAE  (SVM): {mae_svm:.4f}")
print(f"Test R²   (SVM): {r2_svm:.4f}")
print(f"Test MAPE (SVM): {mape_svm:.2f}%")
 
residuals_svm = y_test - y_pred_svm

plt.figure(figsize=(7, 5))
sns.histplot(residuals_svm, bins=40, kde=True, color="teal")
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Residuals (Actual - Predicted)")
plt.title("Residual Distribution – SVM")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_svm, residuals_svm, alpha=0.4, color="darkorange")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Delivery Time (SVM)")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted – SVM")
plt.grid(True)
plt.tight_layout()
plt.show()

# RANDOM FOREST REGRESSOR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# RF pipeline (StandardScaler opsiyoneldir, RF genelde gerektirmez ama pipeline tutarlılığı için eklenmiştir)
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),  # isteğe bağlı
    ("rf", RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_split=2,
        random_state=412,
        n_jobs=-1
    ))
])

# Cross-validated RMSE
cv_rmse_rf = np.sqrt(-cross_val_score(
    rf_pipeline, X_train, y_train,
    scoring="neg_mean_squared_error", cv=cv
))
print(f"CV RMSE (RF): {cv_rmse_rf.mean():.4f}")

# Train the RF model
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

# Performance metrics
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

print(f"Test RMSE (RF): {rmse_rf:.4f}")
print(f"Test MAE  (RF): {mae_rf:.4f}")
print(f"Test R²   (RF): {r2_rf:.4f}")
print(f"Test MAPE (RF): {mape_rf:.2f}%")

# Residuals plot
residuals_rf = y_test - y_pred_rf

plt.figure(figsize=(7, 5))
sns.histplot(residuals_rf, bins=40, kde=True, color="darkgreen")
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Residuals (Actual - Predicted)")
plt.title("Residual Distribution – Random Forest")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_rf, residuals_rf, alpha=0.4, color="forestgreen")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Delivery Time (RF)")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted – RF")
plt.grid(True)
plt.tight_layout()
plt.show()

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# XGBoost pipeline
xgb_pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Normalde gerekli değil ama diğer modellerle tutarlılık için
    ("xgb", XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=412,
        n_jobs=-1
    ))
])

# Cross-validated RMSE
cv_rmse_xgb = np.sqrt(-cross_val_score(
    xgb_pipeline, X_train, y_train,
    scoring="neg_mean_squared_error", cv=cv
))
print(f"CV RMSE (XGBoost): {cv_rmse_xgb.mean():.4f}")

# Train XGBoost
xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)

# Performance metrics
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)

print(f"Test RMSE (XGBoost): {rmse_xgb:.4f}")
print(f"Test MAE  (XGBoost): {mae_xgb:.4f}")
print(f"Test R²   (XGBoost): {r2_xgb:.4f}")
print(f"Test MAPE (XGBoost): {mape_xgb:.2f}%")

# Residuals
residuals_xgb = y_test - y_pred_xgb

# Residual distribution
plt.figure(figsize=(7, 5))
sns.histplot(residuals_xgb, bins=40, kde=True, color="steelblue")
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Residuals (Actual - Predicted)")
plt.title("Residual Distribution – XGBoost")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_xgb, residuals_xgb, alpha=0.4, color="navy")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Delivery Time (XGBoost)")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted – XGBoost")
plt.grid(True)
plt.tight_layout()
plt.show()
