import csv
from sklearn.datasets import make_blobs
import sys 

# Run the command using python3 generatedataset.py num_samples num_centrois num_dimensions 
sampleNumber = sys.argv[1]
centerNumber = sys.argv[2] if ((len(sys.argv)) > 2) else 3 
dimensions = int(sys.argv[3]) if ((len(sys.argv)) > 3) else 2

print(sampleNumber)
X, y, centers = make_blobs(n_samples=int(sampleNumber), centers=int(centerNumber), n_features=int(dimensions), random_state=0, return_centers=True)

with open('dataset.csv', mode='w') as dataset:
    dataset_writer = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(X)):
        dataset_writer.writerow([X[i][j] for j in range(dimensions)]+[y[i]])
    for el in centers:
        dataset_writer.writerow([el[k] for k in range(dimensions)])