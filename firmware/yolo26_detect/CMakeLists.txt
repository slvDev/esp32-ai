cmake_minimum_required(VERSION 3.16)



# Add esp-dl component
# The example is inside esp-dl/examples/yolo26_detect, so esp-dl is at ../../
get_filename_component(ESPDL_PATH "${CMAKE_SOURCE_DIR}/../../" ABSOLUTE)
set(EXTRA_COMPONENT_DIRS "${ESPDL_PATH}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(yolo26_detect)
