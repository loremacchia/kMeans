cmake_minimum_required(VERSION 3.17)
project(kMeans_omp_Branch)

set(CMAKE_CXX_STANDARD 14)

# try also to compile and execute without: -fopenmp
# in Linux just use -fopenmp, without -lomp and -Xpreprocessor
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")

# the following 2 lines are needed on macOS
include_directories(/opt/homebrew/Cellar/libomp/11.0.1/include)
link_directories(/opt/homebrew/Cellar/libomp/11.0.1/lib)

find_package(SFML 2.5 COMPONENTS system window graphics network audio REQUIRED)
include_directories(/opt/homebrew/Cellar/sfml/2.5.1_1/include)

include_directories(omp)

set(CMAKE_CXX_STANDARD 11)


add_executable(kMeans_omp_Branch
        omp/main.cpp
        omp/rapidcsv.h
        dataset.csv
        genDataset.sh
        generateDataset.py
        )


target_link_libraries(kMeans_omp_Branch sfml-system sfml-window sfml-graphics sfml-audio sfml-network)


