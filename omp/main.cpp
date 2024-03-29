#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#ifdef _OPENMP
#include <omp.h> // for OpenMP library functions
#endif
#include <cstdio>
#include <iostream>
#include "rapidcsv.h"
#include <cmath>
#include <chrono>

double* getDataset(int* lenght, int* dim);
double* getFarCentroids(double *points, int pointsLength, int dimensions);


const int k = 3;
int main(int argc, char const *argv[]) {

    int dataLength; // Length of the dataset. It is set by getDataset()
    int dimensions; // Dimension of the dataset. It is set by getDataset()
    double *points = getDataset(&dataLength, &dimensions); // Getting the dataset from the file dataset.csv
    double *centroids = getFarCentroids(points, dataLength, dimensions); // Cluster initialization

    double distanceFromOld = 0; // Variable to chek in the stopping condition. It is the distance of the new set of centroids wrt the old one
    // Representation of a cluster i: centroids[i*dimensions:(i+1)*dimensions-1], pointsInCluster[i], newCentroids[i*dimensions:(i+1)*dimensions-1]
    int pointsInCluster[k]; // Number of points in a cluster i
    double *newCentroids = new double[k*dimensions]; // Temp values of the evaluated new centroids for each cluster


    auto start = std::chrono::system_clock::now();
    // Loop to calculate the final clusters
    do {
        // Init new values
        for (int x = 0; x < k; x++) {
            pointsInCluster[x] = 0;
        }
//#pragma omp parallel for
        for (int x = 0; x < k*dimensions; x++) {
            newCentroids[x] = 0;
        }
        double dist;
        int* clustId = new int[dataLength]; //variable that save the cluster index of each point during iterations of the following for cycle
        double newDist;
        
        //This omp pragma parallelize the following for cycle. dist and newDist become private variables for each running thread because they must be modified atomically by everyone
#pragma omp parallel for num_threads(8) private(dist, newDist)
        for (int i = 0; i < dataLength; i++) {
            dist = 100; // Updated distance from point to the nearest Cluster
            clustId[i] = -1; // Id of the nearest Cluster
            for (int j = 0; j < k; j++) {
                newDist = 0; //Distance from each Cluster
                //#pragma omp parallel for shared(points, centroids) reduction(+: newDist)    //This pragma implements reduction for the calculation of newDist, but it slow down the program for the few iterations of the for
                for (int x = 0; x < dimensions; x++) {
                    newDist += fabs(points[i*dimensions + x] - centroids[j*dimensions + x]);
                }
                if(newDist < dist) {
                    dist = newDist;
                    clustId[i] = j;
                }
            }

        }
        // Assignment of the point to a cluster adding the point's coordinates to the relative centroids 
        // and incrementing the number of points in cluster
        for(int i = 0; i < dataLength; i++) {
            for (int x = 0; x < dimensions; x++) {
                newCentroids[clustId[i] * dimensions + x] += points[i * dimensions + x];
            }
            pointsInCluster[clustId[i]]++;
        }

        // Setting the correct values of newCentroids and updating distanceFromOld and centroids to the actual values
        distanceFromOld = 0;
        for (int j = 0; j < k; j++) {
            for (int x = 0; x < dimensions; x++) {
                newCentroids[j*dimensions + x] /= pointsInCluster[j];
                distanceFromOld += fabs(newCentroids[j*dimensions + x] - centroids[j*dimensions + x]);
                centroids[j*dimensions + x] = newCentroids[j*dimensions + x];
            }
        }
    } while (distanceFromOld > 0.001); // Check stopping condition
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start; // Computation time
    // std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    //Write the computational time into a CSV file
    std::ofstream myfile;
    myfile.open ("./omp/omp.csv", std::ios::app);
    myfile << dataLength;
    printf("\n\nomp: %f\n\n\n",elapsed_seconds.count());
    myfile << "," << elapsed_seconds.count();
    myfile << "\n";
    myfile.close();

    return 0;
}


// Getting the dataset from the CSV file. The last k values are the correct centroids
double* getDataset(int* lenght, int* dim) {
    rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));
    const int rows = int(doc.GetRowCount()) - k;
    const int dimensions = doc.GetColumnCount() - 1;
    *lenght = rows;
    *dim = dimensions;
    // printf("%d\n",rows);
    double *points = new double[rows*dimensions];
    for(int i = 0; i < rows; i++) {
        std::vector<std::string> row = doc.GetRow<std::string>(i);  
        double *array = new double[dimensions];
        int index = 0;
        for(auto element : row) {
            if(index != dimensions) {
                points[i*dimensions + index] = std::atof(element.c_str());
            }
            index++;
        }
    }
    return points;
}

// Centroids initialization function
// The centroids are: a random point from the set (for us the first) and the k-1 furthest points of the set
double* getFarCentroids(double *points, int pointsLength, int dimensions) {
    // Init set of clusters picking a point from the set and the k - 1 points further wrt the chosen point.
    // Those will be the firts clustroids
    double reference[dimensions]; // The first available point is chosen as reference
    for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
        reference[tmpIdx] = points[tmpIdx];
    }
    
    double *distances = new double[k-1]; 
    double *realPoints = new double[k*dimensions]; // Array containing the furthest points wrt reference
    for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
        realPoints[(k-1)*dimensions + tmpIdx] = reference[tmpIdx];
    }
    
    int maxSize = k - 1; // Maximum size of the array and relative indexes
    // Get the k - 1 points
    for(int i = 0; i < pointsLength; i++){
        double dist = 0;
        for (int x = 0; x < dimensions; x++) {
            dist += fabs(points[i*dimensions + x] - reference[x]);
        }
        if(dist > distances[maxSize - 1]) { // If the distance is higher than the last element of the array
            int index = 0;
            while (dist < distances[index] && index < maxSize) { // Find the right place to put in the point
                index++;
            }
            for (int j = maxSize - 1; j > index; j--) {
                distances[j] = distances[j - 1];
                for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
                    realPoints[j*dimensions + tmpIdx] = realPoints[(j - 1)*dimensions + tmpIdx];
                }
            }
            distances[index] = dist;
            for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
                realPoints[index*dimensions + tmpIdx] = points[i*dimensions + tmpIdx];
            }
        }
    }
    return realPoints;
}
