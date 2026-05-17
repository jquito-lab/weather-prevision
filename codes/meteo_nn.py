# -*- coding: utf-8 -*-

import neural_network as neur
import csv
import numpy as np
import matplotlib.pyplot as plt

train_file = "/home/jay/info/GIT/weather-prevision/test3_clean72h.csv"
#%%

def print_tab(tab):
    for elt in tab:
        print(elt)
    return

def calculate_date(str_date):
    # convertit une date sous forme de chaîne en un indice de jour (allant de 0 à 364 en ignorant le 29 février)
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
    # récupère l'écart-type et la moyenne d'une colonne du fichier filename en ignorant les lignes manquantes.
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
    # récupère les données du fichier filename en ignorant les lignes incomplètes,
    # les normalise, et les met sous la forme de jeux de 192 entrées et de jeux de 48 sorties.
    humidity = []
    temp = []
    rain_log = []
    press = []
    day_sin = []
    day_cos = []
    hour_sin = []
    hour_cos = []
    N = 0
    
    with open(filename, mode='r') as file:
        csvfile = csv.reader(file)
        next(csvfile)
        
        for line in csvfile:
            if line[1] != '*':
                N += 1

                #récupération des données
                
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

        # normalisation des données
        
        humidity = np.array(humidity) / 100
            
        temp = (np.array(temp) - mu_T) / sig_T
        press = (np.array(press) - mu_P) / sig_P
        rain_log = (np.array(rain_log) - mu_r) / sig_r

        # mise en forme des jeux d'entrées et de sorties
        
        inputs = [np.concatenate([[day_sin[j], day_cos[j], hour_sin[j], hour_cos[j], temp[j], press[j], humidity[j], rain_log[j]]for j in range(l-23, l+1)]) for l in range(23, N-25)]
        outputs = [np.concatenate([[temp[j], rain_log[j]] for j in range(l+1, l+25)]) for l in range(23, N-25)]
        
    return inputs, outputs

# initialisation du réseau
nn_meteo = neur.create_nn([192, 64, 32, 48])
neur.heetal_init(nn_meteo)

inp, out = get_norm_data_from_file(train_file)

# apprentissage du réseau
neur.grad_descent(nn_meteo, inp, out, 32)



#%% ----------------- Evaluation du réseau ------------------------

test_file = "/home/jay/info/GIT/weather-prevision/codes/test_ann_2026_clean72h.csv"

test_inputs, test_outputs = get_norm_data_from_file(test_file)


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


def evaluate_temp(test_input, test_output):
    norm_prediction = np.array([neur.forward_pass(test_input, nn_meteo)[0][i] for i in range (0, 48, 2)])
    norm_expected = np.array([test_output[i] for i in range(0, 48, 2)])
    
    denorm_prediction = norm_prediction * sig_T + mu_T
    denorm_expected = norm_expected * sig_T + mu_T
    
    for i in range(len(denorm_prediction)):
        plt.plot(denorm_expected[i], denorm_prediction[i], '.b')

    return neur.loss(norm_expected, norm_prediction)


def error_curve_temp():
    C_moy = 0
    n = 0
    for i in range(0, len(test_inputs), 24):
        n += 1
        test_inp = test_inputs[i]
        test_out = test_outputs[i]
        
        C_moy = C_moy + evaluate_temp(test_inp, test_out)

    plt.plot([-10,35], [-10, 35], '-r')
    plt.grid()
    plt.xlabel("Température réelle (°C)")
    plt.ylabel("Température prédite (°C)")
    plt.show()
    C_moy = C_moy/n

    print(f"Coût moyen sur l'échantillon de test pour la température: {C_moy}")
    
    return


def graphs_temp(idx):
    test_input = test_inputs[idx]
    test_output = test_outputs[idx]
    
    norm_prediction = np.array([neur.forward_pass(test_input, nn_meteo)[0][i] for i in range (0, 48, 2)])
    norm_expected = np.array([test_output[i] for i in range(0, 48, 2)])
    
    denorm_prediction = norm_prediction * sig_T + mu_T
    denorm_expected = norm_expected * sig_T + mu_T
    
    plt.plot(range(1,25), denorm_prediction, "-b")
    plt.plot(range(1,25), denorm_expected, "-r")
    plt.legend(["Prédite", "Réelle"])
    plt.xlim((1,24))
    plt.xlabel("Heures après la date "+ idx_to_date(test_file, idx))
    plt.ylabel("Température (°c)")
    plt.grid()
    plt.show()
    
    return


def evaluate_rain(test_input, test_output):
    norm_prediction = np.array([neur.forward_pass(test_input, nn_meteo)[0][i] for i in range (1, 48, 2)])
    norm_expected = np.array([test_output[i] for i in range(1, 48, 2)])
    
    denorm_prediction = np.exp(norm_prediction * sig_r + mu_r) - 1
    denorm_expected = np.exp(norm_expected * sig_r + mu_r) - 1 
    
    

    for i in range(len(denorm_prediction)):
        plt.plot(denorm_expected[i], denorm_prediction[i], '.b')

    return neur.loss(norm_expected, norm_prediction)


def error_curve_rain():
    C_moy = 0
    n = 0
    for i in range(0, len(test_inputs), 24):
        n += 1
        test_inp = test_inputs[i]
        test_out = test_outputs[i]
    
        C_moy = C_moy + evaluate_rain(test_inp, test_out)
    
    plt.plot([-1,10], [-1, 10], '-r')
    plt.xlabel("Précipitations réelles (mm/h)")
    plt.ylabel("Précipitations prédites (mm/h)")
    plt.grid()
    plt.show()
    
    C_moy = C_moy/n
    
    print(f"Coût moyen sur l'échantillon de test pour la pluie: {C_moy}")
    
    return


def graphs_rain(idx):
    test_input = test_inputs[idx]
    test_output = test_outputs[idx]
    norm_prediction = np.array([neur.forward_pass(test_input, nn_meteo)[0][i] for i in range (1, 48, 2)])
    norm_expected = np.array([test_output[i] for i in range(1, 48, 2)])
    
    denorm_prediction = np.exp(norm_prediction * sig_r + mu_r) - 1
    denorm_expected = np.exp(norm_expected * sig_r + mu_r) - 1 
    
    plt.plot(range(1,25), denorm_prediction, "-b")
    plt.plot(range(1,25), denorm_expected, "-r")
    plt.legend(["Prédite", "Réelle"])
    plt.xlim((1,24))
    plt.xlabel("Heures après la date "+ idx_to_date(test_file, idx))
    plt.ylabel("Précipitations tombées en 1h (mm)")
    plt.grid()
    plt.show()
    
    return

def rmse_temp():
    s = 0
    for i in range(len(test_inputs)):
        test_input = test_inputs[i]
        test_output = test_outputs[i]
        
        norm_prediction = np.array([neur.forward_pass(test_input, nn_meteo)[0][i] for i in range (0, 48, 2)])
        norm_expected = np.array([test_output[i] for i in range(0, 48, 2)])
        
        denorm_prediction = norm_prediction * sig_T + mu_T
        denorm_expected = norm_expected * sig_T + mu_T
        
        for j in range(len(denorm_prediction)):
            s = s + (denorm_prediction[j] - denorm_expected[j])**2
    s = s/ (len(test_inputs)*len(denorm_prediction))
    return np.sqrt(s)

def mae_temp():
    s = 0
    for i in range(len(test_inputs)):
        test_input = test_inputs[i]
        test_output = test_outputs[i]
        
        norm_prediction = np.array([neur.forward_pass(test_input, nn_meteo)[0][i] for i in range (0, 48, 2)])
        norm_expected = np.array([test_output[i] for i in range(0, 48, 2)])
        
        denorm_prediction = norm_prediction * sig_T + mu_T
        denorm_expected = norm_expected * sig_T + mu_T
        
        for j in range(len(denorm_prediction)):
            s = s + np.abs(denorm_prediction[j] - denorm_expected[j])
    s = s/ (len(test_inputs)*len(denorm_prediction))
    return s

def rmse_rain():
    s = 0
    for i in range(len(test_inputs)):
        test_input = test_inputs[i]
        test_output = test_outputs[i]
        norm_prediction = np.array([neur.forward_pass(test_input, nn_meteo)[0][i] for i in range (1, 48, 2)])
        norm_expected = np.array([test_output[i] for i in range(1, 48, 2)])
        
        denorm_prediction = np.exp(norm_prediction * sig_r + mu_r) - 1
        denorm_expected = np.exp(norm_expected * sig_r + mu_r) - 1 
        
        for j in range(len(denorm_prediction)):
            s = s + (denorm_prediction[j] - denorm_expected[j])**2
    s = s/ (len(test_inputs)*len(denorm_prediction))
    return np.sqrt(s)

def mae_rain():
    s = 0
    for i in range(len(test_inputs)):
        test_input = test_inputs[i]
        test_output = test_outputs[i]
        norm_prediction = np.array([neur.forward_pass(test_input, nn_meteo)[0][i] for i in range (1, 48, 2)])
        norm_expected = np.array([test_output[i] for i in range(1, 48, 2)])
        
        denorm_prediction = np.exp(norm_prediction * sig_r + mu_r) - 1
        denorm_expected = np.exp(norm_expected * sig_r + mu_r) - 1 
        
        for j in range(len(denorm_prediction)):
            s = s + np.abs(denorm_prediction[j] - denorm_expected[j])
    s = s/ (len(test_inputs)*len(denorm_prediction))
    return s