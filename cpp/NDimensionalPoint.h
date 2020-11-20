#include <iostream>

class NDimensionalPoint {
protected:
    double *pointVector;
    int dimensions = 2;
    int clusterId = -1;

public:
    //default constructor
    NDimensionalPoint(){
        pointVector = new double[dimensions];
        for(int i = 0; i< dimensions; i++){
            pointVector[i] = 0;
        }
    };

    //constructor that performs a deepcopy
    NDimensionalPoint(double *vec, int dim){
        dimensions = dim;
        pointVector = new double[dim];
        for(int i = 0; i< dimensions; i++){
            pointVector[i] = vec[i];
        }
    };

    //copy constructor, also here a deepcopy
    NDimensionalPoint(NDimensionalPoint *p){
        dimensions = p->dimensions;
        pointVector = new double[dimensions];
        for(int i = 0; i< dimensions; i++){
            pointVector[i] = p->pointVector[i];
        }
    };
    
    //delete the point 
    virtual ~NDimensionalPoint(){
        delete[] pointVector;
    };

    double getDistance(NDimensionalPoint p);
    int getDimensions(){return this->dimensions;};
    double* getPoint();
    void setClusterId(int id) {
        clusterId = id;
    }

    void print(){
        for(int i = 0; i < this->dimensions; i++){
            std::cout << pointVector[i] << "  ";
        }
        std::cout << dimensions << "  " << std::endl;
    };
};

