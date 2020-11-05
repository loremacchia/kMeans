#include "NDimensionalPoint.h"
#include <math.h> 

double NDimensionalPoint::getDistance(NDimensionalPoint p){
    double totalDistance = 0;
    for(int i = 0; i < this->dimensions; i++){
        totalDistance = (pointVector[i] - p.pointVector[i])*(pointVector[i] - p.pointVector[i]);
    }
    totalDistance = sqrt(totalDistance);
    return totalDistance;
}


void NDimensionalPoint::addCoordinates(NDimensionalPoint p){
    for(int i = 0; i < this->dimensions; i++){
        pointVector[i] += p.pointVector[i];
    }
}

void NDimensionalPoint::computeDivision(int div){
    for(int i = 0; i < this->dimensions; i++){
        pointVector[i] /= div;
    }
}