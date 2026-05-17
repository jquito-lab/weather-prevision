# -*- coding: utf-8 -*-

import neural_network as neur
import csv
import numpy as np



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

train_file
nn_meteo = neur.create_nn([192, 64, 32, 48])
N = len(nn_meteo["L"])

inputs = [np.concatenate([[day_sin[j], day_cos[j], hour_sin[j], hour_cos[j], temp[j], press[j], humidity[j], rain_log[j]]for j in range(l-23, l+1)]) for l in range(23, N-25)]
outputs = [np.concatenate([[temp[j], rain_log[j]] for j in range(l+1, l+25)]) for l in range(23, N-25)]


neur.heetal_init(nn_meteo)

neur.grad_descent(nn_meteo, inputs, outputs, 200)

