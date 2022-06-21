CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

project(cuda_by_example)

#folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#outpath
SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")  

#Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
if(NOT OpenCV_FOUND)
    message(ERROR " OpenCV not found!")
endif(NOT OpenCV_FOUND)

find_package(CUDA)

CUDA_ADD_EXECUTABLE(ultimatecudadft ultimatecudadft2.cu)
TARGET_LINK_LIBRARIES(ultimatecudadft ${OpenCV_LIBS})
