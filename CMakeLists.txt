cmake_minimum_required(VERSION 2.8)
PROJECT( lsate )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( lsate lsate.cpp )
TARGET_LINK_LIBRARIES( lsate ${OpenCV_LIBS} )