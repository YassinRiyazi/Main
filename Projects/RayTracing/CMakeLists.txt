cmake_minimum_required(VERSION 3.16)
project(raytrace_embree_pcl LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set policies for modern CMake behavior
cmake_policy(SET CMP0167 NEW)
cmake_policy(SET CMP0144 NEW)

# Find required packages
find_package(PCL 1.14 REQUIRED COMPONENTS common io visualization)
find_package(Eigen3 REQUIRED)
find_package(VTK REQUIRED COMPONENTS NO_MODULE)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

# Disable MPI for VTK
set(VTK_MPI_FOUND FALSE CACHE BOOL "Disable MPI for VTK" FORCE)

# Find Embree manually (since it may not have a CMake config)
find_library(EMBREE3_LIBRARY NAMES embree3 libembree3 PATHS /usr/lib /usr/lib/x86_64-linux-gnu)
if(NOT EMBREE3_LIBRARY)
    message(FATAL_ERROR "Embree3 library not found")
endif()

# Include directories
include_directories(${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS} /usr/include/embree3)

# Link directories
link_directories(${PCL_LIBRARY_DIRS} ${VTK_LIBRARY_DIRS} /usr/lib/x86_64-linux-gnu)

# Add definitions for PCL
add_definitions(${PCL_DEFINITIONS})

# Add executable
add_executable(raytrace_embree_pcl raytrace_embree_pcl.cpp)

# Link libraries
target_link_libraries(raytrace_embree_pcl
    PRIVATE
        ${PCL_LIBRARIES}
        ${VTK_LIBRARIES}
        ${EMBREE3_LIBRARY}
        Boost::system
        Threads::Threads
)

# Ensure VTK is properly initialized
if(VTK_FOUND)
    include(${VTK_USE_FILE})
endif()