class DataPoint
{
private:
    
public:
    DataPoint();
    ~DataPoint();
    virtual float getDistance(DataPoint p);
    virtual void addPoint();
    virtual void computeDivision(int div);
};
