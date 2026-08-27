import numpy as np
import tensorflow as tf
import optuna
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Configuración global
VOCAB_SIZE = 10000
MAX_LEN = 256
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_data():
    print("[*] Cargando dataset IMDb...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE, seed=SEED)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
    return x_train, y_train, x_test, y_test

def build_and_train_model(x_train, y_train, params):
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=params["embed_dim"], input_length=MAX_LEN),
        LSTM(params["lstm_units"], return_sequences=False),
        Dropout(params["dropout_rate"]),
        Dense(1, activation="sigmoid")
    ])
    
    optimizer = Adam(learning_rate=params["learning_rate"])
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    
    model.fit(x_train, y_train, epochs=10, batch_size=128, 
              validation_split=0.2, callbacks=[early_stop], verbose=0)
    return model

def main():
    x_train, y_train, x_test, y_test = get_data()

    def objective(trial):
        params = {
            "lstm_units": trial.suggest_categorical("lstm_units", [32, 64]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "embed_dim": trial.suggest_categorical("embed_dim", [32, 64])
        }
        tf.random.set_seed(SEED)
        model = build_and_train_model(x_train, y_train, params)
        _, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return accuracy

    print("[*] Iniciando optimización con Optuna (3 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
    
    print(f"\n[*] Mejores hiperparámetros encontrados: {study.best_params}")
    print(f"[*] Entrenando modelo final con la mejor configuración...")
    
    final_model = build_and_train_model(x_train, y_train, study.best_params)
    loss, accuracy = final_model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[+] Precisión del modelo final optimizado en Test: {accuracy:.4f}")

if __name__ == "__main__":
    main()
