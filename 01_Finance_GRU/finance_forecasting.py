import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error

# Configuración y Semillas
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def create_sequences(data, window_size, future_steps):
    """Crea secuencias de entrada (X) y salida (y) para la predicción."""
    X, y = [], []
    for i in range(len(data) - window_size - future_steps + 1):
        X.append(data[i : (i + window_size)])
        y.append(data[i + window_size : i + window_size + future_steps])
    return np.array(X), np.array(y).reshape(-1, future_steps)

def main():
    print("[*] Descargando datos históricos de AAPL...")
    data = yf.download(tickers="AAPL", start="2020-01-01", end="2025-01-01", auto_adjust=True)
    close_prices = data['Close'].values.reshape(-1, 1)

    # Split y Escalado
    split_index = int(len(close_prices) * 0.8)
    train_data, test_data = close_prices[:split_index], close_prices[split_index:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Hiperparámetros
    WINDOW, FUTURE_STEPS = 20, 3
    X_train, y_train = create_sequences(train_scaled, WINDOW, FUTURE_STEPS)
    X_test, y_test = create_sequences(test_scaled, WINDOW, FUTURE_STEPS)

    print("[*] Construyendo y entrenando el modelo GRU...")
    model = Sequential([
        GRU(64, input_shape=(WINDOW, 1)),
        Dropout(0.2),
        Dense(FUTURE_STEPS)
    ])
    
    model.compile(loss="mean_squared_error", optimizer="adam")
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=32, 
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

    # Predicción y Evaluación
    print("[*] Evaluando el modelo...")
    y_pred_scaled = model.predict(X_test)
    
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, FUTURE_STEPS)
    y_pred_inverse = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1, FUTURE_STEPS)

    for i in range(FUTURE_STEPS):
        rmse = root_mean_squared_error(y_test_inverse[:, i], y_pred_inverse[:, i])
        print(f"  -> RMSE Día {i+1}: {rmse:.2f} USD")

if __name__ == "__main__":
    main()
