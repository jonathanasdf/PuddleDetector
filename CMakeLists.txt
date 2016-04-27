cmake_minimum_required (VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)
project(puddle)

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV 3.1 REQUIRED)
find_package(Boost REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable(PuddleDetector main.cpp)
target_link_libraries(PuddleDetector ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
set_target_properties(PuddleDetector PROPERTIES COMPILE_FLAGS "-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-O3")
