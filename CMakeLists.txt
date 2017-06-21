cmake_minimum_required(VERSION 2.8.8)


project(cudaNormals C CXX)

find_package(Boost REQUIRED)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    message(STATUS "-- Found Boost package" )
else(Boost_FOUND)
    message( SEND_ERROR "-- could not find Boost package" )
endif(Boost_FOUND)

find_package(Threads REQUIRED)

if(Threads_FOUND)
    message(STATUS "-- Found Threads package")
else(Threads_FOUND)
    message( SEND_ERROR "-- could not find Threads package")
endif(Threads_FOUND)

if(LINUX)
    if ( GLUT_Xmu_LIBRARY )
        message( "-- Found X.Org X11 libXmu/libXmuu runtime libraries" )
    else ( GLUT_Xmu_LIBRARY )
        message( SEND_ERROR "-- Could not find X.Org X11 libXmu/libXmuu runtime libraries" )
    endif ( GLUT_Xmu_LIBRARY )
endif(LINUX)

# ---[ misc settings
if(USE_PROJECT_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

##### CUDANORMALS ######



# uncomment of CUDA_TOOLKIT_ROOT_DIR is not found
set(CUDA_TOOLKIT_ROOT_DIR
    /usr/local/cuda/
)


# -- [ this option causes problems for some reason, so turn it off before searching for CUDA
if(MSVC)
	# Setting this to true brakes Visual Studio builds.
	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF CACHE BOOL "CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE")
endif()

# ---[ find dependencies
find_package(CUDA 8.0 REQUIRED)

set(HAVE_CUDA 1)

#--debug
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-G")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-g")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-deviceemu")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11")

#--debug
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_50")

#### CUDA NORMALS CODE #####



include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

# LB Kdtree

set(LB_KDTREE_SRCS
    include/LBKdTree.hpp
    src/LBKdTree.cpp
)

add_library(LbKdTree ${LB_KDTREE_SRCS})

target_link_libraries(LbKdTree
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(LbKdTree PRIVATE cxx_range_for)


# header in srcs because qt-creator cannot find header files
set(NORMALS_CUDA_SRCS
        include/calcNormalsCuda.hpp
        include/PointArray.hpp
        src/calcNormalsCuda.cu
)

#add library to link by calcNormalsCuda.h usage
cuda_add_library(cudaNormals ${NORMALS_CUDA_SRCS})

target_link_libraries(cudaNormals
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        LbKdTree
)


#### EXAMPLE PROGRAM ####



###### ADD YOUR CODE HERE #######

add_executable(NormalsCuda src/main.cpp ext/rply/rply.c)

target_link_libraries(NormalsCuda
	cudaNormals
)

target_compile_features(NormalsCuda PRIVATE cxx_range_for)

add_executable(PoolTest src/pool_test.cpp)

target_link_libraries(PoolTest 
    ${CMAKE_THREAD_LIBS_INIT}    
)

target_compile_features(PoolTest PRIVATE cxx_range_for)
