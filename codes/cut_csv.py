import csv
import numpy as np

test_file = "train_data.csv"
new_file = "half_train.csv"

file_n = 0

lines = []

with open(test_file, mode = 'r') as file:
    csvfile = csv.reader(file)
    next(csvfile)
    
    for line in csvfile:
        file_n += 1
        lines.append(line)

    file_n = file_n // 2
    csvfile = csv.reader(file)

with open(new_file, mode="w") as file:
    writer = csv.writer(file)
    for i in range(file_n):
        writer.writerow(lines[i])

    
    
        
