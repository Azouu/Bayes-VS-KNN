# -*- coding: utf-8 -*-

# CLAUSSE Alexandre, M1 DC.
# LOUAHADJ Inès, M1 DC.

# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import csv

# Types de consonnes
cons_types = ["aa", "ee", "eh", "eu", "ii", "oe", "oh", "oo", "uu", "yy"]

# fichiers à traiter
files = ["data2.csv", "data3.csv", "data12.csv"]

# Récupération des données
def load_dataset(pathname:str):
    """Load a dataset in csv format.

    Each line of the csv file represents a data from our dataset and each
    column represents the parameters.
    The last column corresponds to the label associated with our data.

    Parameters
    ----------
    pathname : str
        The path of the csv file.

    Returns
    -------
    data : ndarray
        All data in the database.
    labels : ndarray
        Labels associated with the data.
    """
    # check the file format through its extension
    if pathname[-4:] != '.csv':
        raise OSError("The dataset must be in csv format")
    # open the file in read mode
    with open(pathname, 'r') as csvfile:
        # create the reader object in order to parse the data file
        reader = csv.reader(csvfile, delimiter=',')
        # extract the data and the associated label
        # (he last column of the file corresponds to the label)
        data = []
        labels = []
        for row in reader:
            data.append(row[:-1])
            labels.append(row[-1])
        # converts Python lists into NumPy matrices
        # in the case of the list of labels, generate an int id per class
        data = np.array(data, dtype=float)
        lookupTable, labels = np.unique(labels, return_inverse=True)
    # return data with the associated label
    return data, labels

def plot_scatter_2d(data,labels) :
    for y in np.unique(labels) :
        x = data[labels == y]
        plt.scatter(x[:,1], x[:,0], s=20, alpha=0.5,  label=cons_types[y])
    plt.legend()
    plt.show()

# Programme principal
if __name__ == '__main__':
    path = './Data'
    X, y = load_dataset(os.path.join(path, files[0]))
    plot_scatter_2d(X,y)



