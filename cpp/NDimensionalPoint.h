class NDimensionalPoint
{
private:
    float *pointVector;
    int dimensions = 2;
    int clusterId = -1;
public:

    NDimensionalPoint(){
        pointVector = new float[0];
    }
    NDimensionalPoint(float *vec, int dim){
        dimensions = dim;
        pointVector = new float[dim];
        for(int i = 0; i< dimensions; i++){
            pointVector[i] = vec[i];
        }
    };

    NDimensionalPoint(NDimensionalPoint *p){
        dimensions = p->dimensions;
        pointVector = new float[dimensions];
        for(int i = 0; i< dimensions; i++){
            pointVector[i] = p->pointVector[i];
        }
    };


    ~NDimensionalPoint(){
        delete[] pointVector;
    };

    float getDistance(NDimensionalPoint p);
    void addCoordinates(NDimensionalPoint p);
    void computeDivision(int div);
    int getDimensions(){return this->dimensions;};
};

