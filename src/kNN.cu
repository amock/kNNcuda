/**
 * kNN.cu
 *
 * @author Alexander Mock
 * @author Matthias Greshake
 */

#include "../include/kNN.h"

struct Matrix {
	int width;
	int height;
	float* elements;
};


__global__ void euclidDist(const Matrix A, Matrix B, int index) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < A.width) {
		B.elements[tid] = (A.elements[tid + 0 * A.width] - A.elements[index + 0 * A.width]) * (A.elements[tid + 0 * A.width] - A.elements[index + 0 * A.width])
                                + (A.elements[tid + 1 * A.width] - A.elements[index + 1 * A.width]) * (A.elements[tid + 1 * A.width] - A.elements[index + 1 * A.width])
                                + (A.elements[tid + 2 * A.width] - A.elements[index + 2 * A.width]) * (A.elements[tid + 2 * A.width] - A.elements[index + 2 * A.width]);
	}
}

__global__ void countElems(const Matrix A, float epsilon, int* result) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < A.width) {
		if (A.elements[tid] < epsilon)
			atomicAdd(result, 1);
	}
}

__global__ void initIndex() {
    d_index = 0;
}

__global__ void getIndices(const Matrix A, Matrix B, float epsilon) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < A.width) {
		if (0 < A.elements[tid] && A.elements[tid] < epsilon) {
                        if (d_index < B.width)
                                B.elements[atomicAdd(&d_index, 1)] = tid;
		}
	}
}


void mallocMat(Matrix &mat) {
	mat.elements = (float*)malloc(mat.width * mat.height * sizeof(float));
}

void initMat(Matrix &mat) {
	for (int i = 0; i < mat.width * mat.height; i++)
		mat.elements[i] = rand() / (float)RAND_MAX * 100;
}

void printMat(const Matrix &mat) {
	for (int i = 0; i < mat.width; i++) {
		for (int j = i; j < mat.width * mat.height; j += mat.width) {
			cout << mat.elements[j];
			if (j < mat.width * (mat.height - 1))
				cout << " | ";
		}
		cout << endl;
	}
}

void getDistances(Matrix &pointMat, Matrix &distVec, int index) {
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (pointMat.width + threadsPerBlock - 1) / threadsPerBlock;

	// Get euclidean distances to each point
	euclidDist<<<blocksPerGrid, threadsPerBlock>>>(pointMat, distVec, index);
}

void evalEpsilon(Matrix &distVec, int* numNeighb, int* d_numNeighb, int k, float &epsilon) {
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (distVec.width + threadsPerBlock - 1) / threadsPerBlock;

	// Learning rate
        float eta = 0.1;
	bool toggle = false;

	do {
		*numNeighb = 0;

		// Count elements below threshold
		cudaMemcpy(d_numNeighb, numNeighb, sizeof(int), cudaMemcpyHostToDevice);
		countElems<<<blocksPerGrid, threadsPerBlock>>>(distVec, epsilon, d_numNeighb);
		cudaMemcpy(numNeighb, d_numNeighb, sizeof(int), cudaMemcpyDeviceToHost);

		// Adapt threshold
		if (*numNeighb > k + 1) {
			epsilon *= 1.0 - eta;
			if (toggle) {
				eta *= 0.5;
				toggle = false;
			}
		}
		else if (*numNeighb < k + 1) {
			epsilon *= 1.0 + eta;
			if (!toggle) {
				eta *= 0.5;
				toggle = true;
			}
		}
	} while (*numNeighb != k + 1);
}

void getNearestNeighbors(Matrix &distVec, Matrix &idxVec, float epsilon) {
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (distVec.width + threadsPerBlock - 1) / threadsPerBlock;

	// Get nearest neighbors
        initIndex<<<1, 1>>>();
        getIndices<<<blocksPerGrid, threadsPerBlock>>>(distVec, idxVec, epsilon);
}

void kNearestNeighborSearch(int k, int numPoints) {
	// Init point matrix at random
	Matrix pointMat;
        pointMat.width = numPoints;
	pointMat.height = 3;
	mallocMat(pointMat);
	initMat(pointMat);

	Matrix idxVec;
	idxVec.width = k;
	idxVec.height = 1;
	mallocMat(idxVec);

        int* numNeighb;
        numNeighb = (int*)malloc(sizeof(int));

	// Write point matrix to device memory
	Matrix d_pointMat;
	d_pointMat.width = pointMat.width;
	d_pointMat.height = pointMat.height;
	size_t size = d_pointMat.width * d_pointMat.height * sizeof(float);
	cudaMalloc(&d_pointMat.elements, size);
	cudaMemcpy(d_pointMat.elements, pointMat.elements, size, cudaMemcpyHostToDevice);

	// Allocate distance vector in device memory
	Matrix d_distVec;
	d_distVec.width = d_pointMat.width;
	d_distVec.height = 1;
	size = d_distVec.width * d_distVec.height * sizeof(float);
	cudaMalloc(&d_distVec.elements, size);

	// Allocate index vector in device memory
	Matrix d_idxVec;
	d_idxVec.width = idxVec.width;
	d_idxVec.height = idxVec.height;
        size = d_idxVec.width * d_idxVec.height * sizeof(float);
        cudaMalloc(&d_idxVec.elements, size);

        // Allocate count variable in device memory
        int* d_numNeighb;
        cudaMalloc(&d_numNeighb, sizeof(int));

	float epsilon = 2500;
	int progress = 0;

	// Iterate over all points
	for (int i = 0; i < pointMat.width; i++) {
		// Calc distances to all other points
		getDistances(d_pointMat, d_distVec, i);

		// Evaluate threshold
                evalEpsilon(d_distVec, numNeighb, d_numNeighb, k, epsilon);

		// Get k nearest neighbors
		getNearestNeighbors(d_distVec, d_idxVec, epsilon);
                cudaMemcpy(idxVec.elements, d_idxVec.elements, size, cudaMemcpyDeviceToHost);

		// Print out progress
                progress = (100 * (i + 1)) / pointMat.width;
                cout << "Progress: " << progress << "%\r";
                cout.flush();
	}

	// Free device memory
        cudaFree(d_numNeighb);
	cudaFree(d_idxVec.elements);
	cudaFree(d_distVec.elements);
	cudaFree(d_pointMat.elements);

        free(numNeighb);
	free(idxVec.elements);
	free(pointMat.elements);
}
