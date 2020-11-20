#include "NDimensionalPoint.h"

class Clustroid : public NDimensionalPoint{
public:
    Clustroid() : NDimensionalPoint() {}
    Clustroid(NDimensionalPoint* point) : NDimensionalPoint(point) {}
    void addCoordinates(NDimensionalPoint p);
    void computeDivision(int div);
};