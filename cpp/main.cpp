#include <iostream>
#include <vector>
#include "rapidcsv.h"

#include "Cluster.h"
#include "NDimensionalPoint.h"
#include "Clustroid.h"


int main(int argc, char const *argv[]) {
    rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));

    int dimensions = doc.GetColumnCount() - 1;
    int rows = int(doc.GetRowCount());
    std::vector<NDimensionalPoint*> points;

    for(int i = 1; i < rows; i++) {
        std::vector<std::string> row = doc.GetRow<std::string>(i);  

        double *array = new double[dimensions];
        int index = 0;
        for(auto element : row) {
            if(index != dimensions) {
                array[index] = std::stof(element);
            }
            index++;
        }
        
        NDimensionalPoint* point = new NDimensionalPoint(array,dimensions);
        points.push_back(point);
    }
    
    // for(auto el : points){
    //     (*el).print();
    // }

    
    int k = 3;
    NDimensionalPoint reference = points[0];

    double *distances = new double[k-1];
    NDimensionalPoint *realPoints = new NDimensionalPoint[k-1];
    int maxIndex = k - 1;

    for(auto el : points){
        double dist = (*el).getDistance(reference);
        if(dist > distances[maxIndex-1]) {
            int i = 0;
            while (dist < distances[i] && i < maxIndex) {
                i++;
            }
            for (int j = maxIndex - 1; j > i; j--) {
                distances[j] = distances[j - 1];
                realPoints[j] = realPoints[j - 1];
            }
            distances[i] = dist;
            realPoints[i] = el;
        }
    }

    std::cout << std::endl << std::endl;
    // for(int i = 0; i < maxIndex; i++){
    //     realPoints[i].print();
    //     std::cout << distances[i] << std::endl;
    // }
    // std::cout << std::endl << std::endl;

    Cluster* clusters[k];
    reference.setClusterId(0);
    clusters[0] = new Cluster(0, &reference);
    clusters[0]->print();
    for (int i = 1; i < k; i++) {
        realPoints[i-1].setClusterId(i);
        clusters[i] = new Cluster(i, &realPoints[i-1]);
        clusters[i]->print();
    }
    std::cout << std::endl << std::endl;
    

    int gigi = 0;
    do {
        for(auto point : points) {
            // (*point).print();
            double dist = 100; // init with a big value TODO check if it is enough
            double newDist = 0;
            int clustId = -1;
            for (int i = 0; i < k; i++) {
                newDist = clusters[i]->getDistance(*point);
                // std::cout << newDist << " vs " << dist << std::endl;
                if(newDist < dist) {
                    dist = newDist;
                    clustId = i;
                }
            }
            clusters[clustId]->addPoint(point);
            point->setClusterId(clustId);

            // (*point).print();
            // std::cout << dist << std::endl << std::endl;
        }    
        for (int i = 0; i < k; i++) {
            std::cout << clusters[i]->newIteration() << std::endl;

            std::cout << std::endl;
            // clusters[i]->print();
        }
        std::cout << std::endl << std::endl;
    } while (++gigi < 2);

    // for(auto el : points){
    //     (*el).print();
    // }
    // std::cout << std::endl << std::endl;




    
}