#pragma once
#include "NDimensionalPoint.cuh"

class Clustroid : public NDimensionalPoint{
public:
    Clustroid() : NDimensionalPoint() {}
    Clustroid(NDimensionalPoint* point) : NDimensionalPoint(point) {}
    __host__ __device__  void addCoordinates(NDimensionalPoint p);
    void computeDivision(int div);
};