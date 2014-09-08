SET(ProjectName MRMS)
PROJECT(${ProjectName})

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMake)


FIND_PACKAGE(OpenMP) 
if (OPENMP_FOUND) 
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OPENMP_FOUND)

IF(UNIX)
#support of ARRAY<T,N>
SET(CMAKE_CXX_FLAGS "-std=c++0x")
ENDIF(UNIX)

# If you don't want to use openCV  and only need KVLD, 
# please commande the next line, 
# don't forget to also commande ${OpenCV_LIBS} further.
FIND_PACKAGE( OpenCV REQUIRED )

#find Orsa and KVLD librarys
ADD_DEFINITIONS(-DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
ADD_SUBDIRECTORY(OrsaHomography)
ADD_SUBDIRECTORY(kvld)

INCLUDE_DIRECTORIES(
./OrsaHomography/
./OrsaHomography/demo/libImage
$ENV{OPENCV_DIR}/install/include/opencv
)

ADD_EXECUTABLE(MRMS_Comparison MRMS_Comparison.cpp convert.h convert.cpp MRMS.h MRMS.cpp)
TARGET_LINK_LIBRARIES(MRMS_Comparison kvld image sift Orsa
${OpenCV_LIBS})


