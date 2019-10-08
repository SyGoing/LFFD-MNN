
#1.cmake verson
cmake_minimum_required(VERSION 2.8)

#2.project name
project(test C CXX)

#opencv
find_package(OpenCV REQUIRED)

#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#4.mnn include
include_directories(${CMAKE_CURRENT_LIST_DIR}/MNN/include)

# mnn lib
add_library( MNN SHARED IMPORTED )
set_target_properties(
        MNN
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/MNN/lib/libMNN.so
)

#6.source directory
file(GLOB TEST_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.h
                    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
set(TEST_COMPILE_CODE ${TEST_SRC})


add_executable(test ${TEST_COMPILE_CODE})

target_link_libraries(test 
                      MNN
                      ${OpenCV_LIBS}
					  )