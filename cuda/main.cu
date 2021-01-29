#include <iostream>
#include <vector>
#include <map> 
#include <string>
#include <chrono>
#include "rapidcsv.h"

#include "Cluster.cuh"
#include "NDimensionalPoint.cuh"
#include "Clustroid.cuh"


const int k = 3;
NDimensionalPoint* getDataset(int* length);
NDimensionalPoint* getFarClustroids(int k, int pointsLength, NDimensionalPoint* points);
__global__ void assignCluster(int length, Cluster* clusters, NDimensionalPoint* points);

int main() {
    int pointsLength = 5;
    NDimensionalPoint* points = getDataset(&pointsLength);
    
    NDimensionalPoint* clustroidPoints = getFarClustroids(k, pointsLength, points);

    for (int i = 0; i < pointsLength; i++) {
        points[i].print();
    }

    // Init the real clusters with the found clustroids
    Cluster* clusters[k];
    for (int i = 0; i < k; i++) {
        clustroidPoints[i].setClusterId(i);
        clusters[i] = new Cluster(i, &clustroidPoints[i]);
    }

    for (int i = 0; i < k; i++) {
        clusters[i]->print();
    }
    /*
    double distFromPrev; // Variable that keeps track for each cluster how far is the new clustroid wrt the previous 
    do {

        assignCluster<<<(pointsLength +127)/128,128>>>(pointsLength, *clusters, points);

        // Compute the new real clustroid for each Cluster and computes the distance from current and previous clustroids
        distFromPrev = 0;
        for (int i = 0; i < k; i++) {
            distFromPrev += clusters[i]->newIteration();
            // clusters[i]->print();
        }
        // std::cout << std::endl;
    } while (distFromPrev > 0); // Stopping condition
*/
}


__global__ void assignCluster(int length, Cluster* clusters, NDimensionalPoint* points){
    if(threadIdx.x < length) {
        int idx = threadIdx.x;
        double dist = 100; // Updated distance from point to the nearest Cluster. Init with a big value. TODO check if it is enough
        double newDist = 0; //Distance from each Cluster
        int clustId = -1; // Id of the nearest Cluster
        for (int i = 0; i < k; i++) {
            newDist = clusters[i].getDistance(points[idx]);
            if(newDist < dist) {
                dist = newDist;
                clustId = i;
            }
        }
        clusters[clustId].addPoint(&(points[idx]));
        points[idx].setClusterId(clustId);
    }
}


NDimensionalPoint* getDataset(int* lenght) {
    rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));
    int dimensions = doc.GetColumnCount() - 1;
    int rows = int(doc.GetRowCount()) - k;
    *lenght = rows;
    printf("%d\n",rows);
    NDimensionalPoint *points = new NDimensionalPoint[10];
    for(int i = 0; i < rows; i++) {
        std::vector<std::string> row = doc.GetRow<std::string>(i);  
        double *array = new double[dimensions];
        int index = 0;
        for(auto element : row) {
            if(index != dimensions) {
                // std::cout << element << std::endl;
                array[index] = std::atof(element.c_str());
            }
            index++;
        }
        NDimensionalPoint* point = new NDimensionalPoint(array,dimensions);
        points[i] = point;
    }
    return points;
}


NDimensionalPoint* getFarClustroids(int k, int pointsLength, NDimensionalPoint* points) {
    // Init set of clusters picking a point from the set and the k - 1 points further wrt the chosen point.
    // Those will be the firts clustroids
    NDimensionalPoint reference = points[0]; // The first available point is chosen as reference
    double *distances = new double[k-1]; 
    NDimensionalPoint* realPoints = new NDimensionalPoint[k]; // Array containing the further points wrt reference
    realPoints[k - 1] = reference;
    int maxSize = k - 1; // Maximum size of the array and relative indexes
    // Get the k - 1 points
    for(int x = 0; x < pointsLength; x++){
        double dist = points[x].getDistance(reference);
        if(dist > distances[maxSize - 1]) { // If the distance is higher than the last element of the array
            int i = 0;
            while (dist < distances[i] && i < maxSize) { // Find the right place to put in the point
                i++;
            }
            for (int j = maxSize - 1; j > i; j--) {
                distances[j] = distances[j - 1];
                realPoints[j] = realPoints[j - 1];
            }
            distances[i] = dist;
            realPoints[i] = points[x];
        }
    }
    return realPoints;
}
