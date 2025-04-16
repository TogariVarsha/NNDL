import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU

# File paths
wheat1_path = "/mnt/data/wheat1.tsv"
wheat599_pc95_tsv_path = "/mnt/data/wheat599_pc95.tsv"

# Load datasets
wheat1_df = pd.read_csv(wheat1_path, sep="\t")
wheat599_pc95_tsv_df = pd.read_csv(wheat599_pc95_tsv_path, sep="\t")

# Merge datasets on 'ID'
df = pd.merge(wheat599_pc95_tsv_df, wheat1_df, on="ID")

# Preprocessing
def preprocess_data(df):
    X = df.drop(columns=["env1", "ID"])  # Features
    y = df["env1"]  # Target
    
    # Normalize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Build Optimized DNNGP Model
def build_optimized_dnngp(input_shape):
    model = Sequential([
        Dense(256, input_shape=(input_shape,)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),

        Dense(128),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),

        Dense(64),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.1),

        Dense(32),
        LeakyReLU(alpha=0.1),
        Dense(1, activation='linear')  # Regression output
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# Train DNNGP
model = build_optimized_dnngp(X_train.shape[1])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Predictions & Evaluation - DNNGP
y_pred_dnn = model.predict(X_test).flatten()
mse_dnn = mean_squared_error(y_test, y_pred_dnn)
r2_dnn = r2_score(y_test, y_pred_dnn)

# Train SVR for Comparison
svr = SVR()
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# Train LightGBM for Comparison
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'learning_rate': 0.05,
    'num_leaves': 31
}
lgb_model = lgb.train(params, lgb_train, valid_sets=[lgb_eval], num_boost_round=100, early_stopping_rounds=10, verbose_eval=False)
y_pred_lgb = lgb_model.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

# Train Random Forest for Comparison
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Results
print("DNNGP Performance: ", "MSE:", mse_dnn, "R²:", r2_dnn)
print("SVR Performance: ", "MSE:", mse_svr, "R²:", r2_svr)
print("LightGBM Performance: ", "MSE:", mse_lgb, "R²:", r2_lgb)
print("Random Forest Performance: ", "MSE:", mse_rf, "R²:", r2_rf)

# Plot Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('DNNGP Training Loss Curve')
plt.legend()
plt.show()

# Scatter Plot of Actual vs. Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_dnn, label='DNNGP', color='blue')
sns.scatterplot(x=y_test, y=y_pred_svr, label='SVR', color='red')
sns.scatterplot(x=y_test, y=y_pred_lgb, label='LightGBM', color='green')
sns.scatterplot(x=y_test, y=y_pred_rf, label='Random Forest', color='purple')
plt.plot(y_test, y_test, color='black', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

# Residual Plot
plt.figure(figsize=(8, 6))
sns.histplot(y_test - y_pred_dnn, bins=30, kde=True, color='blue', label='DNNGP Residuals')
sns.histplot(y_test - y_pred_svr, bins=30, kde=True, color='red', label='SVR Residuals')
sns.histplot(y_test - y_pred_lgb, bins=30, kde=True, color='green', label='LightGBM Residuals')
sns.histplot(y_test - y_pred_rf, bins=30, kde=True, color='purple', label='RF Residuals')
plt.xlabel('Residuals')
plt.title('Residual Distribution')
plt.legend()
plt.show()
