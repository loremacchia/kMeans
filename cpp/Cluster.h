#include "NDimensionalPoint.h"
#include "Clustroid.h"

class Cluster {
private:
    Clustroid* centroid;
    int pointsNumber = 0;
    int clusterId;
public:
    // Constructor of cluster. In copy constructor of NDimensionalPoint is performed a deep copy because here the centroid 
    // is a different point wrt the dataset's points
    Cluster(int id, NDimensionalPoint* initialCentroid){
        clusterId = id;
        centroid = new Clustroid(initialCentroid);
    };

    ~Cluster(){delete centroid;};
    void addPoint(NDimensionalPoint *point);
    Clustroid* divide();
};
