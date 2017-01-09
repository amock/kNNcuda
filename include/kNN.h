/**
 * kNN.h
 *
 * @author Alexander Mock
 * @author Matthias Greshake
 */

#ifndef __KNN_H
#define __KNN_H

#include <cmath>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

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

/**
 * Host functions
 */
void mallocMat(Matrix&);

void initMat(Matrix&);

void printMat(const Matrix&);

void getDistances(Matrix&, Matrix&, int);

void evalEpsilon(Matrix&, int*, int*, int, float&);

void getNearestNeighbors(Matrix&, Matrix&, float);

void kNearestNeighborSearch(int, int);

/**
 * Kernels
 */
__global__ void euclidDist(const Matrix, Matrix, int);

__global__ void countElems(const Matrix, float, int*);

__global__ void initIndex();

__global__ void getIndices(const Matrix, Matrix, float);

#endif // !__KNN_H

