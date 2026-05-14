# -*- coding: utf-8 -*-
#%% ----------------------------Bibliothèques ----------------------------------- #

import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

#%% ----------------- Récupération et normalisation des entrées ------------------- #

train_file = "/home/jay/info/GIT/weather-prevision/codes/small_test.csv"

def calculate_date(str_date):
    month = int(str_date[0:2]) 
    n_days = 0
    for i in range(month-1):
        if i+1 in [1, 3, 5, 7, 8, 10, 12]:
            n_days += 31
        elif i+1 == 2:
            n_days += 28
        else:
            n_days += 30
    day = int(str_date[3:5])
    if month == 2 and day == 29:
        n_days += 28
    else:
        n_days += day
    return n_days-1

def get_mean_std(filename, i_col):
    tab = []
    with open(filename, mode = 'r') as file:
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
        
# Constantes
mu_T, sig_T = get_mean_std(train_file, 3)
mu_P, sig_P = get_mean_std(train_file, 4)
mu_r, sig_r = get_mean_std(train_file, 9)    

def get_norm_data_from_file(filename):
    humidity = []
    temp = []
    rain_log = []
    press = []
    day_sin = []
    day_cos = []
    hour_sin = []
    hour_cos = []
    N = 0
    
    chunks = []

    with open(filename, mode='r') as file:
        csvfile = csv.reader(file)
        next(csvfile)
        
        sep_found = False
        
        humidity = []
        temp = []
        rain_log = []   
        press = []
        day_sin = []
        day_cos = []
        hour_sin = []
        hour_cos = []
        
        n = 0 
        N = 0
        for line in csvfile:
            if line[1] != '*':
                n += 1
                N += 1
                date = calculate_date(line[1][5:10])
                day_sin.append(np.sin(2*np.pi*date/365)) 
                day_cos.append(np.cos(2*np.pi*date/365)) 
                
                hour = float(line[2])
                hour_sin.append(np.sin(2*np.pi*hour/24))
                hour_cos.append(np.cos(2*np.pi*hour/24))    
                
                if line[3] == "":
                    temp.append(temp[-1])
                else:
                    temperature = float(line[3])
                    temp.append(temperature)
                
                if line[4] == "":
                    press.append(press[-1])
                else:   
                    pressure = float(line[4])
                    press.append(pressure)
                
                if line[5] == "":
                    humidity.append(humidity[-1])
                else:
                    hum = float(line[5])
                    humidity.append(hum)
                
                if line[9] == "":
                    rain_log.append(rain_log[-1])
                else:
                    r = float(line[9]) 
                    log_r = np.log(1+r)
                    rain_log.append(log_r)
            else:
                sep_found = True
                humidity = np.array(humidity) / 100
                temp = (np.array(temp) - mu_T) / sig_T
                press = (np.array(press) - mu_P) / sig_P
                rain_log = (np.array(rain_log) - mu_r) / sig_r
                
                chunks.append(np.array([np.array([hour_sin[i], hour_cos[i], day_sin[i], day_cos[i], humidity[i], temp[i], press[i], rain_log[i]]) for i in range(0, n)]))
                
                humidity = []
                temp = []
                rain_log = []   
                press = []
                day_sin = []
                day_cos = []
                hour_sin = []
                hour_cos = []
                n=0
                
        if not(sep_found):
            humidity = np.array(humidity) / 100
            
            temp = (np.array(temp) - mu_T) / sig_T
            press = (np.array(press) - mu_P) / sig_P
            rain_log = (np.array(rain_log) - mu_r) / sig_r
            
            chunks.append(np.array([np.array([hour_sin[i], hour_cos[i], day_sin[i], day_cos[i], humidity[i], temp[i], press[i], rain_log[i]]) for i in range(0, n)]))
    
    return chunks

def idx_to_date(filename, idx):
    with open(filename, mode='r') as file:
        csvfile = csv.reader(file)
        next(csvfile)
        i = 0
        strdate = ""
        for line in csvfile:
            if i == idx:
                strdate = line[1]
                break
            i += 1
    return strdate

#%% -------------------------------- Création du réseau ----------------------------- #

PLUIE_THRESHOLD = 0.5

lookback = 48
n_inputs = 8
window_size = 24

train_chunks = get_norm_data_from_file(train_file)
X_train, Y_rain_train, Y_temp_train = [], [], []


def binarize(tab):
    for i in range(len(tab)):
        if tab[i] > PLUIE_THRESHOLD:
            tab[i] = 1
        else:
            tab[i] = 0
    

for chunk in train_chunks:
    for i in range(lookback, len(chunk)-window_size):
        X_train.append(np.array(chunk[i-lookback:i]))
    for j in range(lookback, len(chunk)-window_size):
        rain_input = np.array([chunk[k][7] for k in range(j, j+window_size)])
        temp_input = np.array([chunk[k][5] for k in range(j, j+window_size)])
        binarize(rain_input)
        Y_rain_train.append(rain_input)
        Y_temp_train.append(temp_input)


X_train = np.array(X_train)
Y_rain_train = np.array(Y_rain_train)
Y_temp_train = np.array(Y_temp_train)

inputs = Input(shape=(lookback, n_inputs))

X = LSTM(64, return_sequences=False)(inputs)
X = Dropout(0.2)(X)
X = Dense(64, activation="relu")(X)

temp_output = Dense(window_size, name="temperature")(X)
rain_output = Dense(window_size, activation="sigmoid", name="rain")(X)

model = Model(inputs= inputs, outputs=[temp_output, rain_output])

model.compile(
        optimizer="adam",
        loss ={
                "temperature": "mse",
                "rain": "binary_crossentropy"
        },
        metrics={
            "temperature": "mae",
            "rain": "accuracy"
        }
    )   

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
    )

model.summary()


#%% --------------------- Entrainement du réseau ---------------------- #
model.fit(
    X_train,
    {"temperature": Y_temp_train,
     "rain": Y_rain_train},
    epochs = 30,
    batch_size = 32,
    validation_split = 0.2,
    callbacks=[early_stop]
    )

#%% --------------------- Importation du réseau ------------------------ #

model = tf.keras.models.load_model("./saved_model.keras", custom_objects=None, compile=True, safe_mode=True)


#%% --------------------- Evaluation du réseau ------------------------- #

test_file = "/home/jay/info/GIT/weather-prevision/codes/today_data.csv"

test_chunks = get_norm_data_from_file(test_file)


X_test, Y_temp_test, Y_rain_test = [], [], []

for chunk in test_chunks:
    for i in range(lookback, len(chunk)-window_size):
        X_test.append(np.array(chunk[i-lookback:i]))
    for j in range(lookback, len(chunk)-window_size):
        rain_input = np.array([chunk[k][7] for k in range(j, j+window_size)])
        temp_input = np.array([chunk[k][5] for k in range(j, j+window_size)])
        binarize(rain_input)
        Y_rain_test.append(rain_input)
        Y_temp_test.append(temp_input)

X_test = np.array(X_test)
Y_rain_test = np.array(Y_rain_test)
Y_temp_test = np.array(Y_temp_test)


losses = model.evaluate(X_test, {"temperature": Y_temp_test, "rain": Y_rain_test})
print(losses)


Y_temp_pred, Y_rain_pred = model.predict(X_test)

def denormalize_temp(tab):
    return tab * sig_T + mu_T

def denormalize_rain(tab):
    tab = tab*sig_r+mu_r
    return np.exp(tab) - 1

Y_temp_pred_real = denormalize_temp(Y_temp_pred)
Y_rain_pred_real = Y_rain_pred
Y_temp_test_real = denormalize_temp(Y_temp_test)

def display_pred_test_temp(idx):
    hours = range(1, window_size+1)
    
    loss = model.evaluate(np.array([X_test[idx]]), {"temperature": np.array([Y_temp_test[idx]]), "rain": np.array([Y_rain_test[idx]])})
    strdate = idx_to_date(test_file, idx)

    plt.figure()
    plt.xlim(0,window_size+1)
    plt.xlabel("x (heures)")
    plt.ylabel("T (°C)")
    plt.ylim(int(min(0, np.min(Y_temp_pred_real[idx]), np.min(Y_temp_test_real[idx]) - 2)), int(5 + max( np.max(Y_temp_pred_real[idx]), np.max(Y_temp_test_real[idx]) )))
    plt.plot(hours, Y_temp_pred_real[idx], '-r', label="Prediction")
    plt.plot(hours, Y_temp_test_real[idx], '-b', label="True")
    plt.title("Prédiction de température à x heures après la date:\n"+ strdate +f"\ntemp_loss = {loss[1]:.2f}\ntemp_mae = {loss[4]:.2f}")
    
    plt.legend()
    plt.grid()
    plt.show()

def display_pred_test_rain(idx):
    hours = range(1,25)
    
    loss = model.evaluate(np.array([X_test[idx]]), {"temperature": np.array([Y_temp_test[idx]]), "rain": np.array([Y_rain_test[idx]])})
    strdate = idx_to_date(test_file, idx)
    
    plt.figure()
    plt.ylim(-0.1,1)
    plt.xlabel("x (heures)")
    plt.ylabel("pluie ou non")
    plt.plot(hours, Y_rain_pred_real[idx], '-r', label="Prediction")
    plt.plot(hours, Y_rain_test[idx], '-b', label="True")
    plt.title("Prédiction des intempéries à x heures après la date:\n"+ strdate +f"\nrain_loss = {loss[2]:.2f}\nrain_bce = {loss[3]:.2f}")
    
    plt.legend()
    plt.grid()
    plt.show()

Y_temp_next, Y_rain_next = model.predict(np.array([test_chunks[0][95:143]]))

Y_temp_next = denormalize_temp(Y_temp_next)
actual_temp = [8.5, 9., 8.9, 8.9, 8.9, 9.2, 8.5, 7.6, 7.7, 9.1, 11.1, 12.3, 12.7, 14., 14.2, 14.5, 14.3, 13.7, 13.7, 12.5, 10.6, 9.1, 8.9, 8.1]

#%% --------------------- Sauvegarde du réseau ------------------------- #

model.save("./saved_model.keras")










