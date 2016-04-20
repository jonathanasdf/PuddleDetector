cmake_minimum_required (VERSION 2.8)
project(puddle)

find_package(PCL 1.7 REQUIRED)
find_package(OpenCV 2.4 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

include_directories("include")

add_executable(main main.cpp)
target_link_libraries(main ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
set_target_properties(main PROPERTIES COMPILE_FLAGS "-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-O3")
