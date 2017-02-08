/**
 * kNN.h
 *
 * @author Alexander Mock
 * @author Matthias Greshake
 */

#ifndef __KNN_H
#define __KNN_H

#include <cmath>
#include <ctime>
#include <iostream>
#include <float.h>
#include <fstream>
#include <sys/time.h>
#include "boost/shared_array.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "../include/rply/rply.h"

using namespace std;

/**
 * Structs
 */
struct Matrix;

/**
 * Global variables
 */
extern int m_mps, m_cuda_cores_per_mp, m_threads_per_mp, m_threads_per_block;

extern int* m_size_block;

extern int* m_size_grid;

extern unsigned long long m_device_global_memory;

__device__ unsigned int d_index = 0;

typedef boost::shared_array<unsigned int> uintArr;

typedef boost::shared_array<float> floatArr;

typedef boost::shared_array<unsigned char> ucharArr;


/**
 * Host functions
 */

/**
 * @brief Allocate memory for matrix
 * @param mat Matrix to be allocate
 */
void mallocMat(Matrix& mat);

/**
 * @brief Initialize matrix with random floats between 0 and 100
 * @param mat Matrix to be initialize
 */
void initMat(Matrix& mat);

/**
 * @brief Print out matrix on the console
 * @param mat Matrix to be print
 */
void printMat(const Matrix& mat);

/**
 * @brief Invoke kernel for calculation of Euclidean distances
 * @param pointMat Source matrix with 3D points
 * @param distVec Result vector with distances
 * @param index Index of the reference point
 */
void getDistances(Matrix& pointMat, Matrix& distVec, int index);

/**
 * @brief Evaluate the threshold that seperates a specific number of points in the neighborhood from 
 *	  the rest of the data based on the Euclidean distance
 * @param distVec Source vector with distances
 * @param numNeighb Counter for the number of neighbors
 * @param d_numNeighb Device counter for the number of neighbors
 * @param k Desired number of points in the neighborhood
 * @param epsilon Initial guess of the threshold
 * @param tol Tolerance for the number of false seperated points
 */
void evalEpsilon(Matrix& distVec, int* numNeighb, int* d_numNeighb, int k, float& epsilon, int tol);

/**
 * @brief Invoke kernel for getting indices of elements in a matrix under a specific threshold
 * @param distVec Source vector with distances
 * @param idxVec Result vector with indices
 * @param epsilon Threshold to separate the data
 */
void getNearestNeighbors(Matrix& distVec, Matrix& idxVec, float epsilon);

/**
 * @brief Calculate direction vectors from on single point to each other point in a matrix
 * @param Source vector with indices
 * @param Source matrix with 3D points
 * @param Result matrix with directions
 * @param index Index of the reference point
 */
void getDirections(const Matrix& idxVec, const Matrix& pointMat, Matrix& neighbMat, int index);

/**
 * @brief Calculate normals from one single point to each other point in a matrix
 * @param neighbMat Source matrix with 3D points
 * @param normalMat Result matrix with normals
 * @param index Index of the reference point
 * @param max_iterations Maximum number of iterations to find normal
 */
void calcNormals(const Matrix& neighbMat, Matrix& normalMat, int index, int max_iterations);

/**
 * I/O-Functions for handling with PLY-files
 */
int readVertexCb(p_ply_argument);

int readColorCb(p_ply_argument);

int readFaceCb(p_ply_argument);

void readPlyFile(Matrix&, const char*, bool, bool, bool, bool, bool);

void writePlyFile(const Matrix&, const Matrix&, const char*);

/**
 * @brief Reduce a point matrix to a specific size by remove points at random
 * @param All_Points Source matrix with 3D points
 * @param V Reduced matrix with 3D points
 * @param target_size Desired size of the matrix
 */
void reducePointCloud(Matrix& All_Points, Matrix& V, int target_size);

/**
 * @brief Find the k nearest neighbors in a 3D point matrix and calculate the normals
 * @param k Desired number of points in the neighborhood
 * @param numPoint Number of 3D points in the matrix
 * @param file PLY-file with 3D points
 * @param dest Destination file to save normals
 */
void kNearestNeighborSearch(int k, int numPoints, const char* file, const char* dest);

/**
 * Kernels
 */

/**
 * @brief Calculate Euclidean distances from one single point to each other point in a matrix
 * @param A Source matrix with 3D points
 * @param B Result vector with distances
 * @param index Index of the reference point
 */
__global__ void euclidDist(const Matrix A, Matrix B, int index);

/**
 * @brief Count elements in a matrix under a specific threshold
 * @param A Source matrix with 3D points
 * @param epsilon Threshold to separate the data
 * @param result Number of elements under the threshold
 */
__global__ void countElems(const Matrix A, float epsilon, int* result);

/**
 * @brief Initialize the index on the device
 */
__global__ void initIndex();

/**
 * @brief Get indices from elements in a matrix under a specific threshold based on the Euclidean 
 *        distance
 * @param A Source vector with distances
 * @param B Result vector with indices
 * @param epsilon Threshold to seperate the data
 */
__global__ void getIndices(const Matrix A, Matrix B, float epsilon);

#endif // !__KNN_H

