# -*- coding: utf-8 -*-

# CLAUSSE Alexandre, M1 DC.
# LOUAHADJ Inès, M1 DC.

# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import csv

# Types de consonnes
cons_types = np.array(["aa", "ee", "eh", "eu", "ii", "oe", "oh", "oo", "uu", "yy"])

# fichiers à traiter
files = np.array(["data2.csv", "data3.csv", "data12.csv"])

# Récupération des données
def load_dataset(pathname:str) -> (np.ndarray, np.ndarray):
    if pathname[-4:] != '.csv':
        raise OSError("The dataset must be in csv format")
    with open(pathname, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        labels = []
        for row in reader:
            data.append(row[:-1])
            labels.append(row[-1])
        data = np.array(data, dtype=float)
        lookupTable, labels = np.unique(labels, return_inverse=True)
    return data, labels

# Affichage des données en 2D
def plot_scatter_2d(X:np.ndarray, y:np.ndarray) -> None:
    plt.scatter(X[:,1], X[:,0], s=20, alpha=0.5, c=y)
    plt.legend(cons_types)
    plt.show()
    
# Affichage d'une matrice de confusion
def plot_confusion_matrix(y_test:np.ndarray, y_pred:np.ndarray, features:np.ndarray, c_name:str, f_name:str) -> None:
    plt.imshow(confusion_matrix(y_test, y_pred))
    plt.xticks(np.arange(len(features)), features, rotation=45)
    plt.yticks(np.arange(len(features)), features)
    plt.ylabel("Véritable classe associée")
    plt.xlabel("Classe prédite")
    plt.title("Matrice de confusion pour {} ({})".format(c_name, f_name))
    plt.colorbar()
    plt.show()

# Programme principal
if __name__ == '__main__':
    path = 'Data'
    for f in range(len(files)):
        X, y = load_dataset(os.path.join(path, files[f]))
        #plot_scatter_2d(X,y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
        #print("X_train: {}".format(X_train))
        #print("X_test: {}".format(X_test))
        #print("y_train: {}".format(y_train))
        #print("y_test: {}".format(y_test))

        # Classification KPPV (KNN)
        n_neighbors = len(cons_types)
        n_test_items = len(X_test)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        knn_predictions = knn.predict(X_test)
        knn_good_predictions = np.sum(y_test == knn_predictions)
        plot_confusion_matrix(y_test, knn_predictions, cons_types, "KPPV", files[f])
        print("Taux de reconnaissance ({}) : {}/{}".format(files[f], knn_good_predictions, n_test_items))
        knn_score = knn_good_predictions / n_test_items
        print("Score de précision associé ({}) : {}".format(files[f], knn_score))
