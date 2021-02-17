import csv
from sklearn.datasets import make_blobs

sampleNumber = 1000
centerNumber = 3
dimensions = 2

X, y, centers = make_blobs(n_samples=sampleNumber, centers=centerNumber, n_features=dimensions, random_state=0, return_centers=True)

# with open('./Implementations/kMeans/dataset.csv', mode='w') as dataset:
with open('dataset.csv', mode='w') as dataset:
    dataset_writer = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(X)):
        dataset_writer.writerow([X[i][0],X[i][1], y[i]])
    for el in centers:
        dataset_writer.writerow([el[0], el[1]])