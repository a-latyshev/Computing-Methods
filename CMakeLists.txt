cmake_minimum_required(VERSION 3.6)
project(MethComp2)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp MethComp2.h MethComp2.cpp)
add_executable(MethComp2 ${SOURCE_FILES})

target_link_libraries(MethComp2 ${ARMADILLO_LIBRARIES})
