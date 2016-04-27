cmake_minimum_required (VERSION 3.1)
project(puddle)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3 -Wno-deprecated-declarations")
include_directories(${PROJECT_SOURCE_DIR})

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV 2.4 REQUIRED)
find_package(Boost REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_definitions(-DEIGEN_MATRIX_PLUGIN="Matrix_initializer_list.h")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable(PuddleDetector main.cpp)
target_link_libraries(PuddleDetector ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
