#include "Clustroid.h"

void Clustroid::addCoordinates(NDimensionalPoint p){
    for(int i = 0; i < this->dimensions; i++){
        pointVector[i] += p.getPoint()[i];
    }
}

void Clustroid::computeDivision(int div){
    for(int i = 0; i < this->dimensions; i++){
        pointVector[i] /= div;
    }
}