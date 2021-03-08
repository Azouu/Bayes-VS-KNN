# -*- coding: utf-8 -*-

# CLAUSSE Alexandre, M1 DC.
# LOUAHADJ Inès, M1 DC.

# Imports
import numpy as np
import os

# Types de consonnes
cons_types = ["aa", "ee", "eh", "eu", "ii", "oe", "oh", "oo", "uu", "yy"]

# fichiers à traiter
files = ["data2.csv", "data3.csv", "data12.csv"]

# Récupération des données
def get_data(path):
    fd = open(path)
    content = fd.read().split("\n")
    for i in range(len(content)):
        if len(content[i]) == 0:
            content.pop(i)
    temp_data = []
    for e in content:
        temp_e = e.split(",")
        temp_data.append(temp_e)
    data = []
    for t in cons_types:
        temp_e = []
        for e in temp_data:
            if e[-1] == t:
                new_e = []
                for i in range(len(e) - 1):
                    new_e.append(float(e[i]))
                temp_e.append(new_e)
        data.append(temp_e)
    return np.array(data)

# Programme principal
if __name__ == '__main__':
    data = get_data(os.path.join("Données", files[0]))
    print(data)
