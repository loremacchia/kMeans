#include "NDimensionalPoint.h"

class Cluster
{
private:
    NDimensionalPoint centroid;
    int pointsNumber = 0;
    int clusterId;
public:
    // Constructor of cluster. In copy constructor of NDimensionalPoint is performed a deep copy because here the centroid 
    // is a different point wrt the dataset's points
    Cluster(int id, NDimensionalPoint initialCentroid){
        clusterId = id;
        centroid = new NDimensionalPoint(&initialCentroid);
    };
    ~Cluster(){delete &centroid;};
};
