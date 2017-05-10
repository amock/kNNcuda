cmake_minimum_required(VERSION 2.8.8)


project(cudaNormals C CXX)

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

#--debug
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_50")

#### CUDA NORMALS CODE #####

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

# header in srcs because qt-creator cannot find header files
set(NORMALS_CUDA_SRCS
        include/calcNormalsCuda.hpp
        include/LBKdTree.hpp
        include/PointArray.hpp
        src/calcNormalsCuda.cu
        src/LBKdTree.cpp
)

#add library to link by calcNormalsCuda.h usage
cuda_add_library(cudaNormals ${NORMALS_CUDA_SRCS})

target_link_libraries(cudaNormals
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
)



#### EXAMPLE PROGRAM ####


find_package(Boost REQUIRED)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    message(STATUS "-- Found Boost package" )
else(Boost_FOUND)
    message( SEND_ERROR "-- could not find Boost package" )
endif(Boost_FOUND)


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

###### ADD YOUR CODE HERE #######
add_executable(Example src/main.cpp include/rply/rply.c)

target_link_libraries(Example
	cudaNormals
)

cuda_add_executable(Test src/cuda_test.cu)

target_link_libraries(Test
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
)
