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

# Taux de distribution des données (test / entrainement)
distribs = np.arange(.1, 1., .1)

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
def plot_confusion_matrix(y_test:np.ndarray, y_pred:np.ndarray, features:np.ndarray, c_name:str, f_name:str, distribution:float, rr1:int, rr2:int) -> None:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.xticks(np.arange(len(features)), features)
    plt.yticks(np.arange(len(features)), features)
    for i in range(len(features)):
        for j in range(len(features)):
            plt.text(i, j, str(cm[i][j]), size='small', ha='center', va='center')
    plt.ylabel("Véritable classe associée (y_gold)")
    plt.xlabel("Classe prédite (y_pred)")
    plt.suptitle("Matrice de confusion du classifieur {}".format(c_name))
    plt.title("{} | d={}/{} | rr={}/{}".format(f_name, int(np.around(distribution*100.,0)), int(np.around((1.-distribution)*100.,0)), rr1, rr2), fontsize=10)
    plt.colorbar()
    plt.show()
    
# Affichage d'un graphe d'évolution du score de précision
def plot_accur_evolution(scores:np.ndarray, distribution:np.array, dims:np.ndarray) -> None:
    plt.figure()
    for e in scores:
        plt.plot(e)
    plt.xticks(np.arange(len(distribution)), [np.around(e, 1) for e in distribution])
    plt.ylabel("Score de précision")
    plt.xlabel("Distribution des données de test")
    plt.legend(dims, title="Dimensions\ndes données")
    plt.title("Evolution du score de précision pour KPPV")
    plt.show()

# Programme principal
if __name__ == '__main__':
    path = 'Data'
    knn_scores = [] 
    dimensions = []
    for f in range(len(files)):
        file_knn_scores = []
        X, y = load_dataset(os.path.join(path, files[f]))
        #plot_scatter_2d(X,y)
        dimensions.append(X.shape[-1])
        print("Avec {} dimensions :".format(dimensions[f]))
        for d in range(len(distribs)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=distribs[d], random_state=0)
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
            plot_confusion_matrix(y_test, knn_predictions, cons_types, "KPPV", files[f], distribs[d], knn_good_predictions, n_test_items)

            # Récupération des scores
            file_knn_score = knn_good_predictions / n_test_items
            file_knn_scores.append(file_knn_score)
        knn_scores.append(file_knn_scores)
        print("---------------------------------------------------------")

    # Courbe d'évolution de la précision pour le classifieur KPPV
    plot_accur_evolution(knn_scores, distribs, dimensions)
