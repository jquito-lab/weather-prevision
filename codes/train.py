# -*- coding: utf-8 -*-
import os
import csv
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from tkinter import Tk, filedialog


# -------------------- Utils paths -------------------- #

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def pick_train_csv():
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choisir le fichier CSV d'entraînement",
        filetypes=[("CSV", "*.csv"), ("Tous les fichiers", "*.*")],
    )
    root.destroy()
    if not path:
        raise RuntimeError("Aucun fichier d'entraînement sélectionné.")
    return path

TRAIN_CSV = pick_train_csv()

TEST_CSV = first_existing([
    os.path.join(BASE_DIR, "test_data.csv"),
    os.path.join(PROJECT_DIR, "test_data.csv"),
])

if TEST_CSV is None:
    raise FileNotFoundError("test_data.csv introuvable (cherché dans codes/ et à la racine du projet).")


# -------------------- Data pipeline (comme nn_tf.py) -------------------- #

def calculate_date(str_date):
    # str_date format: "MM-DD" (dans ton nn_tf: line[1][5:10])
    month = int(str_date[0:2])
    n_days = 0
    for i in range(month - 1):
        if i + 1 in [1, 3, 5, 7, 8, 10, 12]:
            n_days += 31
        elif i + 1 == 2:
            n_days += 28
        else:
            n_days += 30
    day = int(str_date[3:5])
    if month == 2 and day == 29:
        n_days += 28
    else:
        n_days += day
    return n_days - 1


def get_mean_std(filename, i_col):
    tab = []
    with open(filename, mode="r") as file:
        csvfile = csv.reader(file)
        next(csvfile)
        for line in csvfile:
            if line[i_col] == "":
                tab.append(tab[-1])
            else:
                if i_col == 9:
                    tab.append(np.log(1 + float(line[i_col])))
                else:
                    tab.append(float(line[i_col]))
    return np.mean(tab), np.std(tab)


def get_norm_data_from_file(filename, mu_T, sig_T, mu_P, sig_P, mu_r, sig_r, mu_W, sig_W):
    """Retourne (chunks, raw_rain_chunks).

    chunks          : liste de tableaux (N, 12) — features normalisées
    raw_rain_chunks : liste de tableaux (N,)    — pluie brute en mm (binarisation physique)

    Feature layout (12 colonnes) :
        0  hour_sin
        1  hour_cos
        2  day_sin
        3  day_cos
        4  humidity        (/ 100)
        5  temp            (normalisé)
        6  press           (normalisé)
        7  rain_log        (normalisé)
        8  press_tend_3h   (press[t] - press[t-3])           [PRIORITÉ 3]
        9  wind_avg        (normalisé)                        [PRIORITÉ 5]
       10  wind_dir_sin    (sin(2π * dir / 360))              [PRIORITÉ 5]
       11  wind_dir_cos    (cos(2π * dir / 360))              [PRIORITÉ 5]
    """
    chunks = []
    raw_rain_chunks = []
    sep_found = False

    humidity, temp, rain_log, press, rain_raw = [], [], [], [], []
    wind_avg_raw, wind_dir_raw = [], []
    day_sin, day_cos, hour_sin, hour_cos = [], [], [], []
    n = 0

    def _flush(n):
        humidity_arr = np.array(humidity) / 100.0
        temp_arr     = (np.array(temp)         - mu_T) / sig_T
        press_arr    = (np.array(press)        - mu_P) / sig_P
        rain_arr     = (np.array(rain_log)     - mu_r) / sig_r
        wind_arr     = (np.array(wind_avg_raw) - mu_W) / sig_W  # [PRIORITÉ 5]

        # [PRIORITÉ 3] tendance de pression sur 3h
        press_tend = np.zeros(n)
        for i in range(n):
            press_tend[i] = press_arr[i] - press_arr[max(0, i - 3)]

        # [PRIORITÉ 5] direction du vent : variable circulaire → sin/cos
        dir_rad     = np.array(wind_dir_raw) * (2.0 * np.pi / 360.0)
        wind_d_sin  = np.sin(dir_rad)
        wind_d_cos  = np.cos(dir_rad)

        chunk = np.array([
            [hour_sin[i], hour_cos[i], day_sin[i], day_cos[i],
             humidity_arr[i], temp_arr[i], press_arr[i], rain_arr[i], press_tend[i],
             wind_arr[i], wind_d_sin[i], wind_d_cos[i]]
            for i in range(n)
        ], dtype=np.float32)

        return chunk, np.array(rain_raw, dtype=np.float32)

    with open(filename, mode="r") as file:
        csvfile = csv.reader(file)
        next(csvfile)

        for line in csvfile:
            if line[1] != "*":
                n += 1

                date = calculate_date(line[1][5:10])
                day_sin.append(np.sin(2 * np.pi * date / 365))
                day_cos.append(np.cos(2 * np.pi * date / 365))

                hour = float(line[2])
                hour_sin.append(np.sin(2 * np.pi * hour / 24))
                hour_cos.append(np.cos(2 * np.pi * hour / 24))

                if line[3] == "":
                    temp.append(temp[-1])
                else:
                    temp.append(float(line[3]))

                if line[4] == "":
                    press.append(press[-1])
                else:
                    press.append(float(line[4]))

                if line[5] == "":
                    humidity.append(humidity[-1])
                else:
                    humidity.append(float(line[5]))

                if line[9] == "":
                    rain_log.append(rain_log[-1])
                    rain_raw.append(rain_raw[-1] if rain_raw else 0.0)
                else:
                    r = float(line[9])
                    rain_log.append(np.log(1 + r))
                    rain_raw.append(r)

                # [PRIORITÉ 5] vent vitesse + direction
                if line[6] == "":
                    wind_avg_raw.append(wind_avg_raw[-1] if wind_avg_raw else 0.0)
                else:
                    wind_avg_raw.append(float(line[6]))

                if line[7] == "":
                    wind_dir_raw.append(wind_dir_raw[-1] if wind_dir_raw else 0.0)
                else:
                    wind_dir_raw.append(float(line[7]))

            else:
                sep_found = True
                chunk, raw_rain = _flush(n)
                chunks.append(chunk)
                raw_rain_chunks.append(raw_rain)

                humidity, temp, rain_log, press, rain_raw = [], [], [], [], []
                wind_avg_raw, wind_dir_raw = [], []
                day_sin, day_cos, hour_sin, hour_cos = [], [], [], []
                n = 0

        if not sep_found:
            chunk, raw_rain = _flush(n)
            chunks.append(chunk)
            raw_rain_chunks.append(raw_rain)

    return chunks, raw_rain_chunks


def build_xy_from_chunks(chunks, raw_rain_chunks, lookback=48, window_size=24, rain_mm_threshold=0.1):
    """Construit X, Y_temp, Y_rain depuis les chunks.

    [PRIORITÉ 1] Binarisation sur la pluie BRUTE (mm) avec seuil physique (0.1 mm/h),
    plus sur la valeur normalisée log-rain.
    """
    X, Y_rain, Y_temp = [], [], []
    for chunk, raw_rain in zip(chunks, raw_rain_chunks):
        for i in range(lookback, len(chunk) - window_size):
            X.append(chunk[i - lookback:i])  # shape (lookback, 9)

        for j in range(lookback, len(chunk) - window_size):
            # [PRIORITÉ 1] seuil physique sur mm bruts — pas sur la valeur normalisée
            rain_target = np.array(
                [1.0 if raw_rain[k] > rain_mm_threshold else 0.0 for k in range(j, j + window_size)],
                dtype=np.float32,
            )
            temp_input = np.array([chunk[k][5] for k in range(j, j + window_size)], dtype=np.float32)
            Y_rain.append(rain_target)
            Y_temp.append(temp_input)

    return np.array(X, dtype=np.float32), np.array(Y_temp, dtype=np.float32), np.array(Y_rain, dtype=np.float32)


# -------------------- Loss, Metrics & Model -------------------- #

@tf.keras.utils.register_keras_serializable(package="weather")
class WeightedBCE(tf.keras.losses.Loss):
    """[PRIORITÉ 2] BCE pondérée — penalise les faux négatifs pluie de pos_weight fois plus.

    Enregistrée avec @register_keras_serializable pour que model.save() / load_model()
    puissent sérialiser et reconstruire cette loss sans erreur.
    """

    def __init__(self, pos_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = float(pos_weight)

    def call(self, y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weights = y_true * self.pos_weight + (1.0 - y_true)
        return tf.reduce_mean(weights * bce)

    def get_config(self):
        config = super().get_config()
        config["pos_weight"] = self.pos_weight
        return config


@tf.keras.utils.register_keras_serializable(package="weather")
class F1Score(tf.keras.metrics.Metric):
    """[PRIORITÉ 4] F1 score pour classification binaire, calculé depuis Precision + Recall.

    Enregistrée avec @register_keras_serializable pour la sérialisation du modèle.
    """

    def __init__(self, threshold: float = 0.5, name: str = "f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.prec = tf.keras.metrics.Precision(thresholds=threshold, name="_prec")
        self.rec  = tf.keras.metrics.Recall(thresholds=threshold, name="_rec")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.prec.update_state(y_true, y_pred, sample_weight)
        self.rec.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.prec.result()
        r = self.rec.result()
        return 2.0 * p * r / (p + r + tf.keras.backend.epsilon())

    def reset_state(self):
        self.prec.reset_state()
        self.rec.reset_state()

    def get_config(self):
        config = super().get_config()
        config["threshold"] = self.threshold
        return config


def build_model(lookback=48, n_inputs=12, window_size=24, rain_pos_weight=1.0):
    """Architecture avec branche LSTM dédiée à la pluie. [PRIORITÉ 6]

    n_inputs=12 : 9 features précédentes + vent (vitesse + sin/cos direction) [PRIORITÉ 5]
    """
    inputs = Input(shape=(lookback, n_inputs))

    # Encodeur partagé (return_sequences=True pour alimenter les deux branches)
    shared = LSTM(64, return_sequences=True)(inputs)

    # Branche température — LSTM léger
    x_temp = LSTM(32, return_sequences=False)(shared)
    x_temp = Dense(32, activation="relu")(x_temp)
    temp_output = Dense(window_size, name="temperature")(x_temp)

    # Branche pluie — LSTM dédié avec plus de capacité [PRIORITÉ 6]
    x_rain = LSTM(64, return_sequences=False)(shared)
    x_rain = Dropout(0.3)(x_rain)
    x_rain = Dense(64, activation="relu")(x_rain)
    rain_output = Dense(window_size, activation="sigmoid", name="rain")(x_rain)

    model = Model(inputs=inputs, outputs=[temp_output, rain_output])
    model.compile(
        optimizer="adam",
        loss={
            "temperature": "mse",
            "rain": WeightedBCE(pos_weight=rain_pos_weight),
        },
        metrics={
            "temperature": "mae",
            "rain": [  # [PRIORITÉ 4] métriques pertinentes pour classification déséquilibrée
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                F1Score(name="f1"),
            ],
        },
    )
    return model


def main():
    lookback          = 48
    n_inputs          = 12   # 9 précédents + vent (vitesse + sin/cos direction) [PRIORITÉ 5]
    window_size       = 24
    rain_mm_threshold = 0.1  # seuil physique [PRIORITÉ 1]

    # normalisation (calculée sur TRAIN uniquement)
    mu_T, sig_T = get_mean_std(TRAIN_CSV, 3)   # température
    mu_P, sig_P = get_mean_std(TRAIN_CSV, 4)   # pression
    mu_r, sig_r = get_mean_std(TRAIN_CSV, 9)   # rain_log
    mu_W, sig_W = get_mean_std(TRAIN_CSV, 6)   # [PRIORITÉ 5] vitesse du vent

    # save norm params (mu_W/sig_W ajoutés)
    norm_path = os.path.join(ARTIFACTS_DIR, "norm_params.pkl")
    with open(norm_path, "wb") as f:
        pickle.dump((mu_T, sig_T, mu_P, sig_P, mu_r, sig_r, mu_W, sig_W), f)
    print("Norm params saved:", norm_path)

    # build train
    train_chunks, train_raw_rain = get_norm_data_from_file(
        TRAIN_CSV, mu_T, sig_T, mu_P, sig_P, mu_r, sig_r, mu_W, sig_W
    )
    X_train, Y_temp_train, Y_rain_train = build_xy_from_chunks(
        train_chunks, train_raw_rain, lookback=lookback, window_size=window_size, rain_mm_threshold=rain_mm_threshold
    )
    print("X_train:", X_train.shape, "Y_temp_train:", Y_temp_train.shape, "Y_rain_train:", Y_rain_train.shape)

    # [PRIORITÉ 2] calcul du poids de la classe positive (pluie)
    n_pos = float(Y_rain_train.sum())
    n_neg = float((1.0 - Y_rain_train).sum())
    rain_pos_weight = n_neg / max(n_pos, 1.0)
    rain_pct = n_pos / (n_pos + n_neg) * 100
    print(f"Ratio pluie dans Y_rain_train : {rain_pct:.1f}%  →  pos_weight = {rain_pos_weight:.2f}")

    # build & train model
    model = build_model(lookback=lookback, n_inputs=n_inputs, window_size=window_size, rain_pos_weight=rain_pos_weight)
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train,
        {"temperature": Y_temp_train, "rain": Y_rain_train},
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
    )

    # save model
    model_path = os.path.join(ARTIFACTS_DIR, "weather_model.keras")
    model.save(model_path)
    print("Model saved:", model_path)

    # eval sur test
    test_chunks, test_raw_rain = get_norm_data_from_file(
        TEST_CSV, mu_T, sig_T, mu_P, sig_P, mu_r, sig_r, mu_W, sig_W
    )
    X_test, Y_temp_test, Y_rain_test = build_xy_from_chunks(
        test_chunks, test_raw_rain, lookback=lookback, window_size=window_size, rain_mm_threshold=rain_mm_threshold
    )
    if len(X_test) > 0:
        losses = model.evaluate(X_test, {"temperature": Y_temp_test, "rain": Y_rain_test})
        print("Test losses:", losses)

        # [PRIORITÉ 4] rapport F1/Precision/Recall — calcul numpy (pas de dépendance sklearn)
        Y_temp_pred, Y_rain_pred = model.predict(X_test, verbose=0)
        y_pred_bin = (Y_rain_pred > 0.5).astype(int).ravel()
        y_true_bin = Y_rain_test.astype(int).ravel()

        tp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 1)))
        fp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 0)))
        fn = int(np.sum((y_pred_bin == 0) & (y_true_bin == 1)))
        tn = int(np.sum((y_pred_bin == 0) & (y_true_bin == 0)))
        total = tp + fp + fn + tn

        prec_r = tp / (tp + fp + 1e-7)
        rec_r  = tp / (tp + fn + 1e-7)
        f1_r   = 2 * prec_r * rec_r / (prec_r + rec_r + 1e-7)
        prec_s = tn / (tn + fn + 1e-7)
        rec_s  = tn / (tn + fp + 1e-7)
        f1_s   = 2 * prec_s * rec_s / (prec_s + rec_s + 1e-7)

        print("\n=== Rapport pluie (test) ===")
        print(f"{'':>10}  precision  recall  f1-score  support")
        print(f"{'sec':>10}     {prec_s:.2f}    {rec_s:.2f}     {f1_s:.2f}    {tn + fp}")
        print(f"{'pluie':>10}     {prec_r:.2f}    {rec_r:.2f}     {f1_r:.2f}    {tp + fn}")
        print(f"\naccuracy: {(tp + tn) / total:.2%}  ({tp + tn}/{total})")
    else:
        print("Pas assez de données test pour construire X_test.")


if __name__ == "__main__":
    main()