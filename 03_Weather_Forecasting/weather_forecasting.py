import os
import numpy as np
import pandas as pd
import tensorflow as tf
from zipfile import ZipFile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def create_multivariate_sequences(data, history_window, target_window):
    X, y = [], []
    for i in range(len(data) - history_window - target_window + 1):
        X.append(data[i : (i + history_window)])
        # El target asume que la Temperatura es la columna 0
        y.append(data[i + history_window : i + history_window + target_window, 0])
    return np.array(X), np.array(y)

def main():
    print("[*] Descargando y preparando el Jena Climate Dataset...")
    uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    zip_path = tf.keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
    
    with ZipFile(zip_path) as z:
        z.extractall()
        
    df_raw = pd.read_csv("jena_climate_2009_2016.csv")
    features_to_use = ["T (degC)", "p (mbar)", "rh (%)", "wv (m/s)"]
    
    df = df_raw[["Date Time"] + features_to_use].copy()
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df.set_index("Date Time", inplace=True)
    
    # Resampling e interpolación de nulos
    df_hourly = df.resample("60min").mean()
    df_hourly.interpolate(method="linear", inplace=True)
    
    # Split
    split_index = int(len(df_hourly) * 0.7)
    train_df, test_df = df_hourly.iloc[:split_index], df_hourly.iloc[split_index:]
    
    # Escalado (Se guardan dos escaladores: uno para todas las features y otro solo para el target)
    scaler_features = MinMaxScaler()
    scaler_temp = MinMaxScaler()
    
    train_scaled = scaler_features.fit_transform(train_df)
    test_scaled = scaler_features.transform(test_df)
    scaler_temp.fit(train_df[["T (degC)"]])

    HISTORY_STEPS, FUTURE_STEPS = 72, 24
    X_train, y_train = create_multivariate_sequences(train_scaled, HISTORY_STEPS, FUTURE_STEPS)
    X_test, y_test = create_multivariate_sequences(test_scaled, HISTORY_STEPS, FUTURE_STEPS)

    print(f"[*] Entrenando GRU multivariable (In: {X_train.shape[1:]} -> Out: {FUTURE_STEPS}h)...")
    model = Sequential([
        GRU(128, input_shape=(HISTORY_STEPS, X_train.shape[2])),
        Dropout(0.2),
        Dense(FUTURE_STEPS)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=20, batch_size=256, 
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

    print("[*] Evaluando RMSE por horizontes temporales...")
    y_pred_scaled = model.predict(X_test)
    y_test_inverse = scaler_temp.inverse_transform(y_test)
    y_pred_inverse = scaler_temp.inverse_transform(y_pred_scaled)

    horizons = [0, 5, 11, 23] 
    labels = ["+1h", "+6h", "+12h", "+24h"]
    
    for i, h in enumerate(horizons):
        rmse = root_mean_squared_error(y_test_inverse[:, h], y_pred_inverse[:, h])
        print(f"  -> RMSE en {labels[i]}: {rmse:.4f} °C")

if __name__ == "__main__":
    main()
