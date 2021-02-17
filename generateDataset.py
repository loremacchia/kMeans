import csv
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


sampleNumber = 10000
centerNumber = 3
dimensions = 3

X, y, centers = make_blobs(n_samples=sampleNumber, centers=centerNumber, n_features=dimensions, random_state=0, return_centers=True)

with open('dataset.csv', mode='w') as dataset:
    dataset_writer = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(X)):
        dataset_writer.writerow([X[i][j] for j in range(dimensions)]+[y[i]])
    for el in centers:
        dataset_writer.writerow([el[k] for k in range(dimensions)])


