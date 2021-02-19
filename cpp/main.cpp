#include "rapidcsv.h"
#include <math.h> 
#include <chrono> 
#include <vector>
#include <iostream>
#include <fstream>
using namespace std::chrono; 
  


double* getDataset(int* lenght, int* dim);
double* getFarCentroids(double *points, int pointsLength, int dimensions);

const int k = 3;
int main(int argc, char const *argv[]) {
    int dataLength;
    int dimensions;
    double *points = getDataset(&dataLength, &dimensions);
    double *centroids = getFarCentroids(points, dataLength, dimensions);
    std::vector<double> times;
    /*
    for (int i = 0; i < dataLength; i++) {
        for (int j = 0; j < dimensions; j++) {
            printf("%f ", points[i*dimensions + j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dimensions; j++) {
            printf("%f ", centroids[i*dimensions + j]);
        }
        printf("\n");
    }
    */

    double distanceFromOld = 0;
    int pointsInCluster[k]; 
    double *newCentroids = new double[k*dimensions];
    double outerTime = 0;
    do {

        auto start = high_resolution_clock::now(); 
        for (int x = 0; x < k; x++) {
            pointsInCluster[x] = 0;
        }
        for (int x = 0; x < k*dimensions; x++) {
            newCentroids[x] = 0;
        }
        for (int i = 0; i < dataLength; i++) {
            double dist = 100; // Updated distance from point to the nearest Cluster. Init with a big value. TODO check if it is enough
            int clustId = -1; // Id of the nearest Cluster
            for (int j = 0; j < k; j++) {
                double newDist = 0; //Distance from each Cluster
                for (int x = 0; x < dimensions; x++) {
                    newDist += fabs(points[i*dimensions + x] - centroids[j*dimensions + x]);
                }
                if(newDist < dist) {
                    dist = newDist;
                    clustId = j;
                }
            }
            for (int x = 0; x < dimensions; x++) {
                newCentroids[clustId*dimensions + x] += points[i*dimensions + x];
            }
            pointsInCluster[clustId]++;
        }
        distanceFromOld = 0;
        for (int j = 0; j < k; j++) {
            for (int x = 0; x < dimensions; x++) {
                newCentroids[j*dimensions + x] /= pointsInCluster[j];
            }
        }
        for (int j = 0; j < k; j++) {
            for (int x = 0; x < dimensions; x++) {
                distanceFromOld += fabs(newCentroids[j*dimensions + x] - centroids[j*dimensions + x]);
            }
        }
        for (int j = 0; j < k; j++) {
            for (int x = 0; x < dimensions; x++) {
                centroids[j*dimensions + x] = newCentroids[j*dimensions + x];
            }
        }


        auto stop = high_resolution_clock::now(); 

        auto duration = duration_cast<microseconds>(stop - start); 
        double dur = duration.count()/1000;

        times.push_back(dur);
        std::cout << duration.count() << std::endl; 
        outerTime += dur;
        printf("%f",distanceFromOld);
    } while (distanceFromOld > 0.00001);
    printf("\n%f\n",outerTime);

    // int *clusterAssign = new int[dataLength];
    // for (int i = 0; i < dataLength; i++) {
    //     double dist = 100; // Updated distance from point to the nearest Cluster. Init with a big value. TODO check if it is enough
    //     int clustId = -1; // Id of the nearest Cluster
    //     for (int j = 0; j < k; j++) {
    //         double newDist = 0; //Distance from each Cluster
    //         for (int x = 0; x < dimensions; x++) {
    //             newDist += fabs(points[i*dimensions + x] - centroids[j*dimensions + x]);
    //         }
    //         if(newDist < dist) {
    //             dist = newDist;
    //             clustId = j;
    //         }
    //     }
    //     clusterAssign[i] = clustId;
    // }
    
    // std::ifstream oldDataset;
    // oldDataset.open("dataset.csv");

    // std::ofstream newDataset;
    // newDataset.open("./newDataset.csv", std::ios::app);

    // std::string line;
    // int index = 0;
    // while(std::getline(oldDataset, line) && index < dataLength) {
    //     newDataset << line;
    //     newDataset << "," << std::to_string(clusterAssign[index]) << "\n";
    //     index++;
    // }
    
    // oldDataset.close();
    // newDataset.close();

    std::ofstream myfile;
    myfile.open ("cpp.csv", std::ios::app);
    myfile << dataLength;
    myfile << "," << outerTime;
    for(auto element : times) {
        myfile << "," << element;
    }
    myfile << "\n";
    myfile.close();
    return 0;
}




double* getDataset(int* lenght, int* dim) {
    rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));
    const int rows = int(doc.GetRowCount()) - k;
    const int dimensions = doc.GetColumnCount() - 1;
    *lenght = rows;
    *dim = dimensions;
    printf("%d\n",rows);
    double *points = new double[rows*dimensions];
    for(int i = 0; i < rows; i++) {
        std::vector<std::string> row = doc.GetRow<std::string>(i);  
        double *array = new double[dimensions];
        int index = 0;
        for(auto element : row) {
            if(index != dimensions) {
                // std::cout << element << std::endl;
                points[i*dimensions + index] = std::atof(element.c_str());
            }
            index++;
        }
    }
    return points;
}

double* getFarCentroids(double *points, int pointsLength, int dimensions) {
    // Init set of clusters picking a point from the set and the k - 1 points further wrt the chosen point.
    // Those will be the firts clustroids
    double reference[dimensions]; // The first available point is chosen as reference
    for (int tmpIdx = 0; tmpIdx < dimensions; tmpIdx++) {
        reference[tmpIdx] = points[tmpIdx];
    }
    
    double *distances = new double[k-1]; 
    double *realPoints = new double[k*dimensions]; // Array containing the further points wrt reference
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

