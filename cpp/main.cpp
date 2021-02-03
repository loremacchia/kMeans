
#include "rapidcsv.h"
#include <math.h> 

double** getDataset(int* lenght, int* dim);
double** getFarCentroids(double **points, int pointsLength, int dimensions);

const int k = 3;
int main(int argc, char const *argv[]) {
    int dataLength;
    int dimensions;
    double **points = getDataset(&dataLength, &dimensions);
    double **centroids = getFarCentroids(points, dataLength, dimensions);

    double distanceFromOld = 0;
    do {
        int pointsInCluster[k]; 
        double **newCentroids = new double*[k];
        for(int j = 0; j < k; j++) {
            newCentroids[j] = new double[dimensions];
        }
        for (int i = 0; i < dataLength; i++) {
            double dist = 100; // Updated distance from point to the nearest Cluster. Init with a big value. TODO check if it is enough
            int clustId = -1; // Id of the nearest Cluster
            for (int j = 0; j < k; j++) {
                double newDist = 0; //Distance from each Cluster
                for (int x = 0; x < dimensions; x++) {
                    newDist += fabs(points[i][x] - centroids[j][x]);
                }
                if(newDist < dist) {
                    dist = newDist;
                    clustId = j;
                }
            }
            for (int x = 0; x < dimensions; x++) {
                newCentroids[clustId][x] += points[i][x];
                pointsInCluster[clustId]++;
            }
        }
        distanceFromOld = 0;
        for (int j = 0; j < k; j++) {
            for (int x = 0; x < dimensions; x++) {
                newCentroids[j][x] /= pointsInCluster[j];
            }
        }
        for (int j = 0; j < k; j++) {
            for (int x = 0; x < dimensions; x++) {
                distanceFromOld += fabs(newCentroids[j][x] - centroids[j][x]);
            }
        }
        for (int j = 0; j < k; j++) {
            for (int x = 0; x < dimensions; x++) {
                centroids[j][x] = newCentroids[j][x];
            }
        }
        printf("%f\n",distanceFromOld);
    } while (distanceFromOld > 0.000001);
    
    return 0;
}




double** getDataset(int* lenght, int* dim) {
    rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));
    const int rows = int(doc.GetRowCount()) - k;
    const int dimensions = doc.GetColumnCount() - 1;
    *lenght = rows;
    *dim = dimensions;
    printf("%d\n",rows);
    double **points = new double*[rows];
    for(int i = 0; i < rows; i++)
        points[i] = new double[dimensions];
    for(int i = 0; i < rows; i++) {
        std::vector<std::string> row = doc.GetRow<std::string>(i);  
        double *array = new double[dimensions];
        int index = 0;
        for(auto element : row) {
            if(index != dimensions) {
                // std::cout << element << std::endl;
                points[i][index] = std::atof(element.c_str());
            }
            index++;
        }
    }
    return points;
}

double** getFarCentroids(double **points, int pointsLength, int dimensions) {
    // Init set of clusters picking a point from the set and the k - 1 points further wrt the chosen point.
    // Those will be the firts clustroids
    double *reference = points[0]; // The first available point is chosen as reference
    double *distances = new double[k-1]; 
    double **realPoints = new double*[k]; // Array containing the further points wrt reference
    realPoints[k - 1] = reference;
    int maxSize = k - 1; // Maximum size of the array and relative indexes
    // Get the k - 1 points
    for(int i = 0; i < pointsLength; i++){
        double dist = 0;
        for (int x = 0; x < dimensions; x++) {
            dist += fabs(points[i][x] - reference[x]);
        }
        if(dist > distances[maxSize - 1]) { // If the distance is higher than the last element of the array
            int index = 0;
            while (dist < distances[index] && index < maxSize) { // Find the right place to put in the point
                index++;
            }
            for (int j = maxSize - 1; j > index; j--) {
                distances[j] = distances[j - 1];
                realPoints[j] = realPoints[j - 1];
            }
            distances[index] = dist;
            realPoints[index] = points[i];
        }
    }
    return realPoints;
}

