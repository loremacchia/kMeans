#include <iostream>
#include <vector>
#include "rapidcsv.h"
#include "NDimensionalPoint.h"


int main(int argc, char const *argv[])
{
    rapidcsv::Document doc("./dataset.csv", rapidcsv::LabelParams(-1, -1));

    int dimensions = doc.GetColumnCount() - 1;
    // std::cout << dimensions << std::endl;

    std::vector<NDimensionalPoint> points;
    for(int i = 1; i < doc.GetRowCount(); i++) {
        std::vector<std::string> row = doc.GetRow<std::string>(i);  
        double *array = new double[dimensions];
        int index = 0;
        for(auto element : row) {
            if(index != dimensions) {
                array[index] = std::stof(element);
            }
            index++;
        }
        // points.push_back(new NDimensionalPoint(array,dimensions));
    }
    std::cout << points.size() << std::endl;
}