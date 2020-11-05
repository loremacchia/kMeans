#include "NDimensionalPoint.h"

class Cluster
{
private:
    NDimensionalPoint centroid;
    int pointsNumber = 0;
    int clusterId;
public:
    Cluster(int id, NDimensionalPoint initialCentroid){
        clusterId = id;
        centroid = new NDimensionalPoint(&initialCentroid);
    };
    ~Cluster(){delete &centroid;};
};
