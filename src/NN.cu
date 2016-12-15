#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "../include/helper_cuda.h"

// Thread block size
#define BLOCK_SIZE 1
#define NUM 1024

//DRINGEND AUSLAGERN IN HEADER
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

typedef struct {
    int width;
    int height;
    int stride; 
    int* elements;
} MatrixInt;


int m_mps = 0;
int m_cuda_cores_per_mp = 0;
int m_threads_per_mp = 0;
int m_threads_per_block = 0;
int* m_size_thread_block = new int(3);
int* m_size_grid = new int(3);
unsigned long long m_device_global_memory = 0;

using namespace std;

/// std::vector helper methods 

void fillVectorWithRandomFloats(std::vector<float>& mat, int dimX, int dimY) {
	mat.resize(dimX*dimY);
	for(int i=0; i<dimX; i++){
		for(int j=0;j<dimY; j++){
			mat[i*dimY+j] = ((float)rand()/(float)(RAND_MAX)) * 10.0 -5.0;
		}
	}
}

void fillTwoVectorsWithRandomFloats(std::vector<float>& mat1, std::vector<float>& mat2, int dimX, int dimY) {
	mat1.resize(dimX*dimY);
	mat2.resize(dimX*dimY);
	for(int i=0; i<dimX; i++){
		for(int j=0;j<dimY; j++){
			float val = ((float)rand()/(float)(RAND_MAX)) * 10.0 -5.0;
			mat1[i*dimY+j] = val;
			mat2[i*dimX+j] = val;
		}
	}
}





void printVector(std::vector<float>& mat, int dimX, int dimY) {
	mat.resize(dimX*dimY);
	for(int i=0; i<dimX; i++){
		for(int j=0;j<dimY; j++){
			std::cout << mat[i*dimY+j] << " "; 
		}
		std::cout << std::endl;
	}
}


/// Transformation Matrix helper methods

void fill3DTranslationMatrixRightCol(Matrix& m, float x, float y, float z){
	
	//Initial 0
	for(int i=0; i<m.height*m.width; i++)
	{
		m.elements[i] = 0.0;
	}
	
	m.elements[0] = 1.0;
	m.elements[5] = 1.0;
	m.elements[10] = 1.0;
	m.elements[15] = 1.0;
	m.elements[3] = x;
	m.elements[7] = y;
	m.elements[11] = z;
	
}

void fill3DTranslationMatrixDownRow(Matrix& m, float x, float y, float z){
	//Initial 0
	for(int i=0; i<m.height*m.width; i++)
	{
		m.elements[i] = 0.0;
	}
	
	m.elements[0] = 1.0;
	m.elements[5] = 1.0;
	m.elements[10] = 1.0;
	m.elements[15] = 1.0;
	m.elements[12] = x;
	m.elements[13] = y;
	m.elements[14] = z;
}

/// MATRIX helper methods

void mallocMatrix(Matrix& m){
	m.elements = (float*)malloc(m.width * m.height * sizeof(float));
}

void mallocMatrixInt(MatrixInt& m){
	m.elements = (int*)malloc(m.width * m.height * sizeof(int));
}


void fillTwoMatrizesWithRandomFloats(Matrix& m1, Matrix& m2) {
	
	for(int i=0; i<m1.height; i++){
		for(int j=0;j<m1.width; j++){
			float val = ((float)rand()/(float)(RAND_MAX)) * 10.0 -5.0;
			m1.elements[i*m1.width+j] = int(val);
			m2.elements[j*m1.height+i] = int(val);
		}
	}
}

void fillHomogenVectorWithRandomFloats(Matrix& m){
	for(int i=0;i<m.height*m.width-1;i++)
	{
		*(m.elements + i ) = ((float)rand()/(float)(RAND_MAX)) * 10.0 -5.0 ;
		
	}
	*(m.elements + m.height*m.width-1 ) = 1.0;
}

void fillHomogenMatrixWithRandomFloats(Matrix& m1){
	for(int i=0; i<m1.height-1; i++){
		for(int j=0;j<m1.width; j++){
			float val = ((float)rand()/(float)(RAND_MAX)) * 10.0 -5.0;
			m1.elements[i*m1.width+j] = val;
		}
		
	}
	for(int j=(m1.height-1)*m1.width;j<m1.width*m1.height; j++){
		m1.elements[j] = 1.0;
	}
}

void transposeMatrix(Matrix& m1, Matrix& m2){
	
	for(int i=0; i<m1.height; i++){
		for(int j=0;j<m1.width; j++){
			m2.elements[j*m1.height+i] = m1.elements[i*m1.width+j];
		}
	}
}


void fillMatrixWithRandomFloats(Matrix& m)
{
	int i;
	//int j;
	for(i=0;i<m.height*m.width;i++)
	{
		*(m.elements + i ) = (((float)rand()/(float)(RAND_MAX)) * 10.0 - 5.0);
	}
}

void printMatrix(Matrix& m)
{
	int i;
	//int j;
	for(i=0;i<m.width*m.height;i++)
	{
		if(i%m.width == 0){
			printf("|");
		}
		printf(" %f ",*(m.elements + i ));
		if(i%m.width == m.width-1){
			printf("|\n");
		}
	}
	
	printf("\n");
}

void printMatrixInt(MatrixInt& m)
{
	int i;
	//int j;
	for(i=0;i<m.width*m.height;i++)
	{
		if(i%m.width == 0){
			printf("|");
		}
		printf(" %d ",*(m.elements + i ));
		if(i%m.width == m.width-1){
			printf("|\n");
		}
	}
	
	printf("\n");
}

void getColVecOfMatrix(const Matrix& m, int index , Matrix& v_out){
	for(int i=0;i< m.height;i++){
		v_out.elements[i] = m.elements[index+i*m.width];
	}
}


// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
//TODO: test
__global__ void SelfScalarKernel(const Matrix, Matrix);
//TODO: implement
__global__ static void MergeSort(Matrix m, Matrix results);
//TODO: implement
__global__ static void SortKernel(Matrix m, int limit=-1);

__global__ static void combSortKernel(Matrix m);

__global__ static void combSortKernel2(Matrix m, int gap, bool* test_bool=NULL);

__global__ static void evaluateEpsilonKernel(Matrix m, float epsilon, int* num_results = NULL);

__global__ static void SelectionKernel(Matrix m, MatrixInt dest, float epsilon);

__global__ static void InitSelectionKernel();

__global__ void DistanceKernel(const Matrix A, const Matrix B, Matrix dest);

__global__ void DistanceKernel2(const Matrix A, int index, Matrix dest);

__global__ static void SortKernel2(Matrix m, int* slide_buffer, int num_slides, int limit = -1, int offset=0);

__global__ static void SelectColKernel(Matrix m, Matrix dest, int index);

	

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix& A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix& A, int row, int col) 
{

    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}




// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(Matrix& A, Matrix& B, Matrix& C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);         
            
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);

    cudaMalloc(&d_B.elements, (int)size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

	clock_t calcstart, calcend;
	calcstart = clock();
	
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	calcend=clock();
	printf("Multiplikation %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);
	
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);


    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}


//first transform
void SelfScalar(Matrix& A, Matrix& C)
{
	// Load A to device memory
	Matrix d_A;
	d_A.width = d_A.stride = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
	       cudaMemcpyHostToDevice);         

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = d_C.stride = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	clock_t calcstart, calcend;
	calcstart = clock();

	// Invoke kernel
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (A.width +threadsPerBlock-1)/threadsPerBlock;

	SelfScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C);

	calcend=clock();
	printf("Inner product %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);

	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size,
	       cudaMemcpyDeviceToHost);


	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_C.elements);
}

void SelectColGPU(Matrix& D_V, Matrix& D_nn_point, int index, int width){

	SelectColKernel<<<1, 1>>>(D_V, D_nn_point, index);

}


void DistancesGPU2(Matrix& D_V, Matrix& D_distance_vec, int index, int width)
{
	  
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (width +threadsPerBlock-1)/threadsPerBlock;

	DistanceKernel2<<<blocksPerGrid, threadsPerBlock>>>(D_V, index, D_distance_vec);

}

void DistancesGPU(Matrix& D_V, Matrix& D_nn_point, Matrix& D_distance_vec, int width)
{
	  
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (width +threadsPerBlock-1)/threadsPerBlock;

	DistanceKernel<<<blocksPerGrid, threadsPerBlock>>>(D_V, D_nn_point, D_distance_vec);

}

// without transform
void Distances(Matrix& A, Matrix& B, Matrix& C)
{
	// Load A to device memory
	Matrix d_A;
	d_A.width = d_A.stride = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
	       cudaMemcpyHostToDevice);

	
	Matrix d_B;
	d_B.width = d_B.stride = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,
	       cudaMemcpyHostToDevice);         

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = d_C.stride = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	//~ clock_t calcstart, calcend;
	//~ calcstart = clock();

	// Invoke kernel
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (A.width +threadsPerBlock-1)/threadsPerBlock;

	//printf("bpg %d , tpb %d\n",blocksPerGrid,threadsPerBlock);

	DistanceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

	//~ calcend = clock();
	//printf("Distance calculation %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);

	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size,
	       cudaMemcpyDeviceToHost);

	
	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

void combSort2(Matrix& A) {
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    clock_t calcstart, calcend;
    calcstart = clock();

	bool* D_ResultVector;

	cudaMalloc((void**)&D_ResultVector, sizeof(bool) );
	
	bool* H_ResultVector = (bool*)malloc(sizeof(bool));


	int gap = A.width;

    bool sorted = false;
	
    while(!sorted) {
		gap /= 1.3;
        if(gap < 1){
			gap = 1;
            sorted = true;
		}

		int threadsPerBlock = m_threads_per_block;
		int blocksPerGrid = (A.width + threadsPerBlock - 1) / (gap *2 );

		//std::cout << "sorting with gap: " << gap << std::endl;
		combSortKernel2<<<blocksPerGrid, threadsPerBlock>>>(d_A, gap, D_ResultVector);//, D_ResultVector);
		cudaMemcpy(H_ResultVector, D_ResultVector, sizeof(bool), cudaMemcpyDeviceToHost);
		
		
		sorted = (*(H_ResultVector));
		//std::cout << "Sorted: " << sorted << std::endl;
		
        if(gap > 1) {
            sorted = false;
        } 
		
    }



    calcend = clock();
    printf("Sorting %f milliseconds\n",(float)(calcend-calcstart) * 1000.0 / CLOCKS_PER_SEC);

    cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A.elements);
    cudaFree(D_ResultVector);
    
    free(H_ResultVector);
}

float EvaluateEpsilonGPU(Matrix& D_distance_vec, int width, int k, float epsilon, int* H_ResultVector, int* D_ResultVector, int tolerance = 0){
	
	
	
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (width +threadsPerBlock-1)/threadsPerBlock;
	*H_ResultVector = 0;
	
	
	//~ clock_t calcstart, calcend;
    //~ calcstart = clock();
    
    float change_factor = 0.1;
    // epsilon *= (1+0.1)
    // epsilon *= (1-0.1)
    bool first_value = true;
    bool epsilon_under = false;
    
        while( *H_ResultVector < k + 1 || *H_ResultVector > k + 1 + tolerance ) {
	
		*H_ResultVector = 0;
		//num_results = 0;

	
		
		
		//~ clock_t calcstart, calcend;
		//~ calcstart = clock();
		
		cudaMemcpy(D_ResultVector, H_ResultVector, sizeof(int), cudaMemcpyHostToDevice);
		
		
		evaluateEpsilonKernel<<<blocksPerGrid, threadsPerBlock>>>(D_distance_vec, epsilon, D_ResultVector);
	
		
		cudaMemcpy(H_ResultVector, D_ResultVector, sizeof(int), cudaMemcpyDeviceToHost);
		
		//~ calcend = clock();
		//~ printf("Eval Iteration time GPU %f milliseconds \n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);
		
		
	
		//printf("found %d results for epsilon %f. k is %d\n", *H_ResultVector, epsilon, k);
		
                if( *H_ResultVector > k+1+tolerance){
			epsilon *= (1.0-change_factor);
			if(first_value){
				//printf("first value is over limit\n");
				epsilon_under = false;
				first_value = false;
			}else if(epsilon_under){
				change_factor*=0.5;
				//printf("over limit: change_factor changed to %f\n",change_factor);
				epsilon_under = false;
			}
                } else if(*H_ResultVector < k+1) {
			epsilon *= (1.0+change_factor);
			if(first_value){
				//printf("first value is under limit\n");
				epsilon_under = true;
				first_value = false;
			}else if(!epsilon_under){
				change_factor*=0.5;
				//printf("under limit: change_factor changed to %f\n",change_factor);
				epsilon_under = true;
			}
		}
	
	}
	
	//~ printf("\n");
		
		
	
	
	return epsilon;
}

float evaluateEpsilon(Matrix& A, int k, float epsilon, int& num_distances, int tolerance = 0){
	Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	
	int* D_ResultVector;
	cudaMalloc((void**)&D_ResultVector, sizeof(int) );
	
	int* H_ResultVector = (int*)malloc(sizeof(int));
	
	
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (A.width +threadsPerBlock-1)/threadsPerBlock;
	*H_ResultVector = 0;
	
	
	//~ clock_t calcstart, calcend;
    //~ calcstart = clock();
    
    float change_factor = 0.2;
    bool first_value = true;
    bool epsilon_under = false;
    
	while( *H_ResultVector < k || *H_ResultVector > k + tolerance ) {
	
		*H_ResultVector = 0;
		

	
		cudaMemcpy(D_ResultVector, H_ResultVector, sizeof(int), cudaMemcpyHostToDevice);
	
		evaluateEpsilonKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, epsilon, D_ResultVector);
	
		cudaMemcpy(H_ResultVector, D_ResultVector, sizeof(int), cudaMemcpyDeviceToHost);
	
		//printf("found %d results for epsilon %f. k is %d\n", *H_ResultVector, epsilon, k);
		
		if( *H_ResultVector > k+tolerance){
			epsilon *= (1.0-change_factor);
			if(first_value){
				//printf("first value is over limit\n");
				epsilon_under = false;
				first_value = false;
			}else if(epsilon_under){
				change_factor*=0.5;
				//printf("over limit: change_factor changed to %f\n",change_factor);
				epsilon_under = false;
			}
		}else if(*H_ResultVector < k){
			epsilon *= (1.0+change_factor);
			if(first_value){
				//printf("first value is under limit\n");
				epsilon_under = true;
				first_value = false;
			}else if(!epsilon_under){
				change_factor*=0.5;
				//printf("under limit: change_factor changed to %f\n",change_factor);
				epsilon_under = true;
			}
		}
	
	}
	num_distances = *H_ResultVector;
	//~ calcend = clock();
    //~ printf("Epsilon Evaluation %f milliseconds\n",(float)(calcend-calcstart) * 1000.0 / CLOCKS_PER_SEC);

	
	cudaFree(d_A.elements);
	cudaFree(D_ResultVector);
	free(H_ResultVector);
	return epsilon;
}



void GetDistancesUnderEpsilonGPU(Matrix& D_distance_vec, MatrixInt& D_indices_vec, int width, float epsilon){

    
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (width + threadsPerBlock-1)/threadsPerBlock;
    
    InitSelectionKernel<<<1,1>>>();
    SelectionKernel<<<blocksPerGrid, threadsPerBlock>>>(D_distance_vec, D_indices_vec, epsilon);
	
	
    
}

void getDistancesUnderEpsilon(Matrix& A, MatrixInt& B, float epsilon){
	Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t sizeA = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, sizeA);
    cudaMemcpy(d_A.elements, A.elements, sizeA, cudaMemcpyHostToDevice);
    
    MatrixInt d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size_t sizeB = B.width * B.height * sizeof(int);
    cudaMalloc(&d_B.elements, sizeB);
    
    
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (A.width + threadsPerBlock-1)/threadsPerBlock;
    
    InitSelectionKernel<<<1,1>>>();
    SelectionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, epsilon);
	cudaMemcpy(B.elements, d_B.elements, sizeB, cudaMemcpyDeviceToHost);
	
    
    
    cudaFree(d_A.elements);
	cudaFree(d_B.elements);
}


void combSort(Matrix& A) {
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    clock_t calcstart, calcend;
    calcstart = clock();

    int threadsPerBlock = m_threads_per_block;
    int blocksPerGrid = (A.width + threadsPerBlock - 1) / threadsPerBlock;

    combSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A);

    calcend = clock();
    printf("Sorting %f milliseconds\n",(float)(calcend-calcstart) * 1000.0 / CLOCKS_PER_SEC);

    cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A.elements);
}


void Sort(Matrix& A, Matrix& B)
{
	// Load A to device memory
	Matrix d_A;
	d_A.width = d_A.stride = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);         

	// Load A to device memory
	Matrix d_B;
	d_B.width = d_B.stride = A.width; d_B.height = A.height;
	cudaMalloc(&d_B.elements, size);
	//cudaMemcpy(d_B.elements, A.elements, size, cudaMemcpyHostToDevice);         
    
        
	
	
	

	clock_t calcstart, calcend;
	calcstart = clock();
	// Invoke kernel
	
	//~ int threadsPerBlock = m_threads_per_block;
	//~ int blocksPerGrid = (A.width +threadsPerBlock-1)/threadsPerBlock;
	//~ MergeSort<<<1,A_size>>>(d_A,d_B);
	
	//~ int A_size = A.width*A.height;
	//MergeSort<<<1, A_size, sizeof(float)*A_size*2>>>(d_A,d_B);
	//~ int sort_iterations = log(A_size)/log(2);
	//~ std::cout << sort_iterations << std::endl;
	//~ for(int i=0;i<sort_iterations;i++)
	//~ {
	int A_size = A.width*A.height;
	SortKernel<<<1,  A_size, sizeof(float) * A_size * 2 >>>(d_A);
	//~ }


	calcend=clock();
	printf("Sorted %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);

	// Read C from device memory
	//cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(B.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
}

//second try
void Sort2(Matrix& m, Matrix& dest, int limit=-1){
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (m.width +threadsPerBlock-1)/threadsPerBlock;

	printf("bpg %d , tpb %d\n",blocksPerGrid,threadsPerBlock);
	
	Matrix d_m;
	d_m.width = d_m.stride = m.width; d_m.height = m.height;
	size_t m_size = m.width * m.height * sizeof(float);
	cudaMalloc(&d_m.elements, m_size);
	cudaMemcpy(d_m.elements, m.elements, m_size, cudaMemcpyHostToDevice);         

	
	int m_elements = m.width*m.height;
	int slide_buffer_size = int(m_elements-0.5);
	int* slide_buffer = (int*) malloc(slide_buffer_size * sizeof(int));

	clock_t calcstart, calcend;
	calcstart = clock();

	//create RUNS
	int num_slides=1;
	slide_buffer[0] = 0;
	for(int x=1; x < slide_buffer_size+1; x++) {
		if(m.elements[x] < m.elements[x-1])
		{
			slide_buffer[num_slides] = x;
			num_slides++;
		}
	}
	slide_buffer[num_slides] = m_elements;
	slide_buffer_size = num_slides+1;
	
	// Load A to device memory
	int* d_slide_buffer;
	int slide_buffer_mem_size = slide_buffer_size* sizeof(int);
	cudaMalloc(&d_slide_buffer, slide_buffer_mem_size );
	cudaMemcpy(d_slide_buffer, slide_buffer, slide_buffer_mem_size, cudaMemcpyHostToDevice);  
	

	//sort 
	int count = 0;
	int current_limit = -1;
	while(num_slides > 1){
		
	
		if(num_slides > 2){
			current_limit = limit;
		}
		
		//int dim = m.width*m.height;
		
		//cudaMemcpy(slide_buffer, d_slide_buffer, slide_buffer_size * sizeof(int), cudaMemcpyHostToDevice);
		//SortKernel2<<<1,  dim, sizeof(float) * dim * 2 >>>(d_m,d_slide_buffer, num_slides, current_limit);
		
		
		//int partition = threadsPerBlock*blocksPerGrid ;
		
		//printf("awda %d\n",blocksPerGrid);
		
		
		cudaMemcpy(slide_buffer, d_slide_buffer, (num_slides+1) * sizeof(int), cudaMemcpyHostToDevice);
		SortKernel2<<<blocksPerGrid, threadsPerBlock >>>(d_m,d_slide_buffer, num_slides, current_limit);
		cudaMemcpy(slide_buffer, d_slide_buffer, (num_slides+1) * sizeof(int), cudaMemcpyDeviceToHost);
		
		
		
		
		//SortKernel2<<<blocksPerGrid, threadsPerBlock >>>(d_m,d_slide_buffer, num_slides, current_limit,10);
		
		//cudaMemcpy(slide_buffer, d_slide_buffer, slide_buffer_size * sizeof(int), cudaMemcpyDeviceToHost);
		
		count ++;
		num_slides = int(num_slides/2.0+0.5);
		
	}
	
	calcend = clock();
	printf("Sort GPU %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);

	
	cudaMemcpy(dest.elements, d_m.elements, m_size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_m.elements);
	cudaFree(d_slide_buffer);
	free(slide_buffer);
}

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

//first distance function after transformation
__global__ void SelfScalarKernel(Matrix A, Matrix Dest)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    	if (i < Dest.height * Dest.width)
    	{
		float dest=0.0;
		int j;
		for(j=0;j<A.height;j++)
		{
			 dest += A.elements[j*A.width+i] * A.elements[j*A.width+i];
		}
		Dest.elements[i] = dest-1;
	}
}

__global__ static void SelectColKernel(Matrix m, Matrix dest, int index){
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid == 0){
		for(int i=0;i< m.height;i++){
			dest.elements[i] = m.elements[index + i * m.width];
		}
	}
}

//distance function without transformation
__global__ void DistanceKernel(const Matrix points,const Matrix s_point, Matrix dest)
{
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	//~ if(tid>20000)
		//~ printf("thread: %d\n",tid);
	
	
	int num_points = points.width;
	if(tid < num_points)
	{
		dest.elements[tid] = (points.elements[tid + 0 * points.width] - s_point.elements[0]) * (points.elements[tid + 0 * points.width] - s_point.elements[0]) 
								+ (points.elements[tid + 1 * points.width] - s_point.elements[1]) * (points.elements[tid + 1 * points.width] - s_point.elements[1])
								+ (points.elements[tid + 2 * points.width] - s_point.elements[2]) * (points.elements[tid + 2 * points.width] - s_point.elements[2]) ; 
	}
}

//distance function without transformation
__global__ void DistanceKernel2(const Matrix points,int index, Matrix dest)
{
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(tid < points.width)
	{
		dest.elements[tid] = (points.elements[tid + 0 * points.width] - points.elements[index+ 0* points.width]) * (points.elements[tid + 0 * points.width] - points.elements[index+ 0* points.width] ) 
								+ (points.elements[tid + 1 * points.width] - points.elements[index+ 1* points.width]) * (points.elements[tid + 1 * points.width] - points.elements[index+ 1* points.width] )
								+ (points.elements[tid + 2 * points.width] - points.elements[index+ 2* points.width]) * (points.elements[tid + 2 * points.width] - points.elements[index+ 2* points.width] ) ; 
	}
}

__device__ inline void Merge2(float* a, int i1, int j1, int i2, int j2,int limit=-1){
	
	int limit_copy = limit;
	
	float* temp = (float*) malloc((j2-i1+1) * sizeof(float));  //array used for merging
    int i,j,k;
    i=i1;    //beginning of the first list
    j=i2;    //beginning of the second list
    k=0;
    
    int counter = 0;
    while(i<=j1 && j<=j2 && limit!=0 )    //while elements in both lists
    {
		counter ++;
		limit --;
        if(a[i]<a[j])
            temp[k++]=a[i++];
        else
            temp[k++]=a[j++];
    }
    
    while(i<=j1 && limit!=0)    //copy remaining elements of the first list
        temp[k++]=a[i++];
        limit--;
        
    while(j<=j2 && limit!=0)    //copy remaining elements of the second list
        temp[k++]=a[j++];
        limit--;
        
    //Transfer elements from temp[] back to a[]
    for(i=i1,j=0;i<=j2 && limit_copy!=0;i++,j++,limit_copy--)
	{
        a[i] = temp[j];
    }   
    free(temp);
}

__device__ inline void Merge(float* values, float* results, int l, int r, int u)
{
	int i,j,k;
	i=l; j=r; k=l;
	while(i<r && j<u){
		if(values[i]<=values[j])
		{
			results[k] = values[i];
			i++;
		}else{
			results[k] = values[j];
			j++;
		}
		k++;
	}
	
	while(i<r){
		results[k] = values[i];
		i++;
		k++;
	}
	
	while(j<u){
		results[k] = values[j];
		j++;
		k++;
	}
	
	for(k=l;k<u;k++){
		values[k]=results[k];
	}
	
}

__global__ static void MergeSort(Matrix m, Matrix results){
	
	extern __shared__ float shared[];
	
	//const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int tid =  threadIdx.x;
	//const unsigned int tid = blockIdx.x;
	int k,u,i;
	
	int m_size = m.width*m.height;
	
	
	shared[tid] = m.elements[tid];
	
	__syncthreads();
	
	k=1;
	while(k < m_size)
	{
		i=0;
		while(i+k<m_size)
		{
			u = i+k*2;
			if(u > m_size)
			{
				u = m_size;
			}
			//printf("%d %d %d \n",i,k,u);
			Merge(shared,results.elements, i,i+k, u);
			i = i+k*2;
		}
		k = k*2;
		__syncthreads();
		//printf("Iter %d\n",i);
	}
	
	m.elements[tid] = shared[tid];
}

__global__ static void SortKernel2(Matrix m, int* slide_buffer, int num_slides, int limit , int offset){
	
	//const unsigned int i =  threadIdx.x * 2;
	const unsigned int i = (blockDim.x * blockIdx.x + threadIdx.x)*2 + offset;
			
	
	if(i>4000){
		//printf("thread %d\n",i);		
	}
	if( i>=2 && i < num_slides+1 ) 
	{
		Merge2(m.elements, slide_buffer[i-2], slide_buffer[i-1]-1, slide_buffer[i-1], slide_buffer[i]-1,limit);
	}
	__syncthreads();
	
	if(i>=2 && i < num_slides+1 ) 
	{
		slide_buffer[i/2]= slide_buffer[i];
	}
	
	__syncthreads();
	
	if(num_slides%2 == 1 && i == offset){
		slide_buffer[(num_slides+1)/2] = slide_buffer[num_slides];
	}
	
	
}


__global__ static void combSortKernel2(Matrix m, int gap, bool* test_bool) {
	
	
	*test_bool = true;
	
	
	
	const unsigned int tid = gap *2 * blockIdx.x + threadIdx.x;
	// es gehen (thread_size - gap) threads verloren  
	if( threadIdx.x < gap  && tid + gap < m.width)
	{
		//printf("%d * %d + %d = %d\n",gap * 2,blockIdx.x,threadIdx.x, tid);
		
        if(m.elements[tid] > m.elements[tid+gap]) 
        {
			//printf("m.elements[%d]=%f > m.elements[%d]=%f\n",tid,m.elements[tid],tid+gap,m.elements[tid+gap]);
            //printf("switched %d, %d\n",tid,tid+gap);
            
            float tmp = m.elements[tid];
            m.elements[tid] = m.elements[tid+gap];
            m.elements[tid+gap] = tmp;
            *test_bool = false;
            
        }
	}  
	
	
	
}

__global__ static void combSortKernel(Matrix m) {
    int gap = m.width;
    bool sorted = false;

    while(!sorted) {
        gap = gap / 1.3;
        //gap = gap / (1+0.3/2048.0);
        if(gap > 1) {
            sorted = false;
        } else {
            gap = 1;
            sorted = true;
        }

        const unsigned int tid = blockDim.x*2 * blockIdx.x + threadIdx.x;
		if(blockIdx.x <= 2)
			printf("%d * %d + %d = %d\n",blockDim.x * 2,blockIdx.x,threadIdx.x, tid);
		
        if(tid + gap < m.width) {
            if(m.elements[tid] > m.elements[tid+gap]) {
                float tmp = m.elements[tid];
                m.elements[tid] = m.elements[tid+gap];
                m.elements[tid+gap] = tmp;
                sorted = false;
            }
        }
        __syncthreads();
    }
}



__global__ static void evaluateEpsilonKernel(Matrix m, float epsilon, int* num_results){
	
	//~ __shared__ int num_results_shared;
	//~ num_results_shared = 0;
	//~ __syncthreads();
	
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < m.width){
		if( m.elements[tid] < epsilon){
			atomicAdd(num_results,1);
		}
	}
	
	//~ __syncthreads();
	//~ *num_results = num_results_shared;
	
}

__device__ unsigned int select_index = 0;



__global__ static void InitSelectionKernel(){
	select_index = 0;
}

__global__ static void SelectionKernel(Matrix m, MatrixInt dest, float epsilon){
	
	
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < m.width){
                if( 0 < m.elements[tid] && m.elements[tid] < epsilon){
			if(select_index < dest.width){
				dest.elements[ atomicAdd(&select_index,1)  ] = tid;
			}
		}
	}
	
	
	
	
}





__global__ static void SortKernel(Matrix m, int limit){
	
	int m_elements = m.width*m.height;
	
	int slide_buffer_size = int(m_elements-0.5);
	
	extern __shared__ int buffer[];
	extern __shared__ float values[];
	__shared__ int check_runs;
	__shared__ int num_runs;
	
	
	check_runs=0;
	
	int* slide_buffer = (int*) malloc(slide_buffer_size * sizeof(int));

	//create RUNS
	int num_slides=1;
	slide_buffer[0] = 0;
	for(int x=1; x < slide_buffer_size && check_runs==0; x++) {
		if(m.elements[x] < m.elements[x-1])
		{
			slide_buffer[num_slides] = x;
			buffer[num_slides] = x;
			num_slides++;
		}
		
	}
	__syncthreads();
	if(check_runs == 0){
		check_runs=1;
		num_runs = num_slides;
		
		slide_buffer[num_slides] = m_elements;
		slide_buffer_size = num_slides+1;
		
	
		//sort 
		int count = 0;
		int current_limit = -1;
		while(num_slides > 1){
			//__syncthreads();
			if(num_slides <= 2){
				current_limit = limit;
			}
			const unsigned int i =  threadIdx.x;
			//printf("%d\n",i);
			
			if(i>=2 && i<int(num_slides+1) && i%2==0 ) 
			{
				printf("numslides %d , index %d\n",num_slides,i);
				//parallelisierbar
				//__syncthreads();
				Merge2(m.elements, slide_buffer[i-2], slide_buffer[i-1]-1, slide_buffer[i-1], slide_buffer[i]-1,current_limit);
				//~ Merge2(m.elements, buffer[i-2], buffer[i-1]-1, buffer[i-1], buffer[i]-1,current_limit);
				
			}
			
			__syncthreads();
			
			if(i>=2 && i<int(num_slides+1) && i%2 == 0 ) 
			{
				slide_buffer[i/2-1]= slide_buffer[i-2];
				slide_buffer[i/2]= slide_buffer[i];
			}
			
			__syncthreads();
		
			
			if(num_slides%2 == 1){
				slide_buffer[(num_slides+1)/2] = slide_buffer[num_slides];
			}
			
			count ++;
			num_slides = int(num_slides/2.0+0.5);
			
			
		}
		
	}
	
	
	free(slide_buffer);
}

void getCudaInformation(int& mps, int& cuda_cores_per_mp, int& threads_per_mp, int& threads_per_block, int* size_thread_block, int* size_grid , unsigned long long& device_global_memory){
	cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    
    mps = deviceProp.multiProcessorCount;
    cuda_cores_per_mp = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    threads_per_mp = deviceProp.maxThreadsPerMultiProcessor;
    threads_per_block = deviceProp.maxThreadsPerBlock;
    size_thread_block[0] = deviceProp.maxThreadsDim[0];
    size_thread_block[1] = deviceProp.maxThreadsDim[1];
    size_thread_block[2] = deviceProp.maxThreadsDim[2];
    size_grid[0] = deviceProp.maxGridSize[0];
    size_grid[1] = deviceProp.maxGridSize[1];
    size_grid[2] = deviceProp.maxGridSize[2];
    device_global_memory = (unsigned long long) deviceProp.totalGlobalMem;
    
}

void mergeHost(float* a, int i1, int j1, int i2, int j2,int limit=-1){
	int limit_end = limit;
	
	
	float* temp = (float*) malloc((j2-i1+1) * sizeof(float));  //array used for merging
    int i,j,k;
    i=i1;    //beginning of the first list
    j=i2;    //beginning of the second list
    k=0;
    
    int counter = 0;
    while(i<=j1 && j<=j2 && limit!=0)    //while elements in both lists
    {
		counter ++;
		limit--;
        if(a[i]<a[j])
            temp[k++]=a[i++];
        else
            temp[k++]=a[j++];
    }
    
    while(i<=j1 && limit!=0)    //copy remaining elements of the first list
        temp[k++]=a[i++];
        
    while(j<=j2 && limit!=0)    //copy remaining elements of the second list
        temp[k++]=a[j++];
        
    //Transfer elements from temp[] back to a[]
    for(i=i1,j=0;i<=j2 && limit_end!=0 ;i++,j++,limit_end--)
	{
        a[i] = temp[j];
    }   
    free(temp);
}


void combSortSerial(Matrix& m) {
    int gap = m.width;
    bool sorted = false;

    while(!sorted) {
        gap = gap / 1.3;
        cout << gap << endl;
        if(gap > 1) {
            sorted = false;
        } else {
            gap = 1;
            sorted = true;
        }

        int i = 0;
        float tmp = 0.0;
        while(i + gap < m.width) {
            if(m.elements[i] > m.elements[i+gap]) {
                tmp = m.elements[i];
                m.elements[i] = m.elements[i+gap];
                m.elements[i+gap] = tmp;
                sorted = false;
            }
            i++;
        }
    }
}

void naturalMergeSort(Matrix& m, int limit=-1){
	int m_elements = m.width*m.height;
	
	int slide_buffer_size = int(m_elements-0.5);
	int* slide_buffer = (int*) malloc(slide_buffer_size * sizeof(int));

	clock_t calcstart, calcend;
	calcstart = clock();

	//create RUNS
	int num_slides = 1;
	slide_buffer[0] = 0;
	for(int i=1; i < slide_buffer_size; i++) {
		if(m.elements[i] < m.elements[i-1])
		{
			slide_buffer[num_slides] = i;
			num_slides++;
		}
		
	}
	slide_buffer[num_slides] = m_elements;
	slide_buffer_size = num_slides+1;
	
	
	//sort 
	int count = 0;
	int current_limit = -1;
	while(num_slides > 1){
		if(num_slides > 2){
			current_limit = limit;
		}
		std::cout << count+1 <<" Iteration: You can use " << int(num_slides/2) << " Threads" << std::endl;
		int i;
		
		for(i=2;i<int(num_slides+1);i+=2)
		{
			//parallelisierbar
			mergeHost(m.elements, slide_buffer[i-2], slide_buffer[i-1]-1, slide_buffer[i-1], slide_buffer[i]-1,current_limit);
			
			slide_buffer[i/2-1]= slide_buffer[i-2];
			slide_buffer[i/2]= slide_buffer[i];
		}
		
		if(num_slides%2 == 1){
			slide_buffer[(num_slides+1)/2] = slide_buffer[num_slides];
		}
		
		count ++;
		num_slides = int(num_slides/2.0+0.5);
		
	}
	
	calcend = clock();
	printf("Sort CPU %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);
	
	
	free(slide_buffer);
}

int main2(int argc, char** argv)
{
        getCudaInformation(m_mps, m_cuda_cores_per_mp, m_threads_per_mp, m_threads_per_block, m_size_thread_block, m_size_grid, m_device_global_memory);
	std::cout << std::endl;
	std::cout << "Device Information" << std::endl;
	std::cout << "mps: " << m_mps << std::endl;
	std::cout << "cuda_cores_per_mp: " << m_cuda_cores_per_mp << std::endl;
	std::cout << "threads_per_mp: " << m_threads_per_mp << std::endl;
	std::cout << "threads_per_block: " << m_threads_per_block << std::endl;
	std::cout << "size_thread_block: " << m_size_thread_block[0] << ", " << m_size_thread_block[1] << ", "<< m_size_thread_block[2]  << std::endl;
	std::cout << "size_grid: " << m_size_grid[0] << ", " << m_size_grid[1] << ", "<< m_size_grid[2]  << std::endl;
	std::cout << "device_global_memory: " << m_device_global_memory << std::endl;
	std::cout << std::endl;
	
	//plan: (T * v_points) * (T * v_points)^T 
	int seed = 1479731956;
	//int seed = time(NULL);
	printf("%d\n",seed);
    	srand(seed);
    
	//point vector ALLE PUNKTE
	// V muss auf GPU geladen werden: 1mal
	Matrix V;
	V.height = 3;
        V.width = 5000000;
	std::cout << "points " << V.width << std::endl; 
	V.stride = V.width;
	mallocMatrix(V);
	fillMatrixWithRandomFloats(V);
	//printMatrix(V);
	
	// soll nur auf der GPU angelegt werden
	//point for searchings
	Matrix nn_point;
	nn_point.height = 3;
	nn_point.width = 1;
	nn_point.stride = nn_point.width;
	mallocMatrix(nn_point);
	
	
	// soll nur auf der GPU angelegt werden
	Matrix distance_vec;
	distance_vec.height = 1;
	distance_vec.width = V.width;
	distance_vec.stride = distance_vec.width;
	mallocMatrix(distance_vec);
	
	int k = 10;
	//int index = 20000;
	float initial_epsilon = 0.003792;
	
	// soll auf der GPU und Arbeitsspeicher angelegt werden
	// am ende in 
	MatrixInt indices_vec;
	indices_vec.height = 1;
	//indices_vec.width = num_shortes_distances;
	indices_vec.width = k;
	indices_vec.stride = indices_vec.width;
	mallocMatrixInt(indices_vec);
	//point index for searching
	
	float epsilon = initial_epsilon;
	printf("START\n");
	for(int i = 0;i<20;i++){
		
		printf("\n%d\n",i);
		//methode nur mit GPU MAtrizen
		getColVecOfMatrix(V,i,nn_point);
		//~ std::cout << "pic point " << index << std::endl;
	
		// Methode nur mit GPU Matrizen
		Distances(V, nn_point, distance_vec);
		
		//~ printMatrix(distance_vec);
		
    
		int num_shortes_distances = 0;
    
		// distance vec soll GPU matrix sein
		epsilon = evaluateEpsilon( distance_vec, k, epsilon, num_shortes_distances);
		//~ printf("epsilon found: %f, num_distances: %d\n",epsilon, num_shortes_distances);
		printf("epsilon: %f\n",epsilon);
    
		// distance_vec soll nur gpu matrize sein , indices_vec kommt zurzueck
		// epsilon wird ausserhalb der gpu uebertragen
		// 
		getDistancesUnderEpsilon(distance_vec,indices_vec,epsilon);
		
		//~ printMatrixInt(indices_vec);
    }
    printf("END\n");
    
    //getDistancesUnderEpsilon(distance_vec,indices_vec,epsilon);
	printMatrixInt(indices_vec);
	
	std::cout << "FREE" << std::endl;
	//~ free(unsortedVector.elements);
	free(V.elements);
	free(distance_vec.elements);
	free(indices_vec.elements);
	free(nn_point.elements);
	
	
	
	/*
	 * 1. Auf Arbeitsspeicher: V,indices_vec
	 * 2. Auf GPU: D_V, D_nn_point, D_distance_vec, D_indices_vec
	 * 3. V wird in D_V kopiert
	 * 4. fuer jedes i
	 * 		a. index i wird zur GPU geschickt
	 * 		b. Auf GPU:
	 * 			speichere punkt D_V[i] in D_nn_point
	 * 			Schneeschaufel: 
	 * 			rechne mit D_V und D_nn_point Distanzen aus -> D_distance_vec
	 * 			Scheiß auf Sortierung Sortierung:
	 * 			schicke epsilon(initial) von CPU zu GPU evaluiere epsilon mit k -> epsilon
	 * 			schicke epsilon von CPU zu GPU und bekomme k naechste nachbarn in D_indices_vec
	 * 		c. Kopiere D_indices_vec zu Host -> indices_vec
	 * 		
	 */
	
	return 0;
}

int readNumPointsFromFile(const char* filename){
	return 50;
}

void loadPointsFromFile(const char* filename, Matrix& V){
	
} 

void generateHostMatrix(MatrixInt& m, int width, int height){
	
	m.height = height;
	m.width = width;
	m.stride = m.width;
	mallocMatrixInt(m);
	
}

void generateHostMatrix(Matrix& m, int width, int height){
	
	m.height = height;
	m.width = width;
	m.stride = m.width;
	mallocMatrix(m);
	
}

void generateDeviceMatrix(MatrixInt& D_m, int width, int height){
	
	D_m.width = D_m.stride = width;
    D_m.height = height;
    size_t size = D_m.width * D_m.height * sizeof(int);
    cudaMalloc(&D_m.elements, size);
    
}

void generateDeviceMatrix(Matrix& D_m, int width, int height){
	
    D_m.width = D_m.stride = width;
    D_m.height = height;
    size_t size = D_m.width * D_m.height * sizeof(float);
    cudaMalloc(&D_m.elements, size);
    
}

void generateDeviceMatrices(Matrix& D_V,Matrix& D_nn_point,Matrix& D_distance_vec, MatrixInt& D_indices_vec, int width, int height, int k){
	
	generateDeviceMatrix(D_V, width, height);
	generateDeviceMatrix(D_nn_point, 1 , height);
	generateDeviceMatrix(D_distance_vec, width, 1);
	generateDeviceMatrix(D_indices_vec, k, 1);
	
}

void copyToDeviceMatrix(Matrix& m, Matrix& D_m){
	
	size_t size = m.width * m.height * sizeof(float);
    cudaMemcpy(D_m.elements, m.elements, size, cudaMemcpyHostToDevice);

}

void copyToDeviceMatrix(MatrixInt& m, MatrixInt& D_m){
	
	size_t size = m.width * m.height * sizeof(int);
    cudaMemcpy(D_m.elements, m.elements, size, cudaMemcpyHostToDevice);

}

void copyToHostMatrix(Matrix& D_m, Matrix& m){
	
	size_t size = m.width * m.height * sizeof(float);
	cudaMemcpy(m.elements, D_m.elements, size, cudaMemcpyDeviceToHost);
	
}

void copyToHostMatrix(MatrixInt& D_m, MatrixInt& m){
	
	size_t size = m.width * m.height * sizeof(int);
	cudaMemcpy(m.elements, D_m.elements, size, cudaMemcpyDeviceToHost);
	
}

float kNNGpu(Matrix& D_V, Matrix& D_nn_point, Matrix& D_distance_vec, MatrixInt& D_indices_vec, int* H_ResultVector, int* D_ResultVector,
					int width , int k, int index, float epsilon, int DEBUG = false)
{
	
	clock_t calcstart, calcend;
	if(DEBUG){
		calcstart = clock();
	}
	
	//SelectColGPU(D_V,D_nn_point,index,width);
	
	if(DEBUG){
		calcend = clock();
		printf("select GPU %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);
	
	
		calcstart = clock();
	}
	//~ printf("%d\n",*H_ResultVector);
	DistancesGPU2(D_V,D_distance_vec,index,width);
	
	if(DEBUG){
		calcend = clock();
		printf("Distances GPU %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);
	
		calcstart = clock();
	}
	
	
	epsilon = EvaluateEpsilonGPU(D_distance_vec,width,k,epsilon,H_ResultVector,D_ResultVector,5);
	
	if(DEBUG){
		calcend = clock();
		printf("evaluate epsilon GPU %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);
	
		calcstart = clock();
	}
	
	
	GetDistancesUnderEpsilonGPU(D_distance_vec,D_indices_vec,width,epsilon);
	
	if(DEBUG){
		calcend = clock();
		printf("get nearest points GPU %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);
	}
	
	return epsilon;
	
}

void freeCudaMatrices(Matrix& D_V, Matrix& D_nn_point, Matrix& D_distance_vec, MatrixInt& D_indices_vec){
	
	cudaFree(D_V.elements);
	cudaFree(D_nn_point.elements);
	cudaFree(D_distance_vec.elements);
	cudaFree(D_indices_vec.elements);
	
}

int main(int argc, char** argv)
{
	getCudaInformation(m_mps, m_cuda_cores_per_mp, m_threads_per_mp, m_threads_per_block, m_size_thread_block, m_size_grid, m_device_global_memory);
	
	
	int k=50;
	
	const char * filename  = "points.ply";
	
	// Host Matrices
	Matrix V;
	MatrixInt indices_vec;
	
	// Device Matrices
	Matrix D_V;
	Matrix D_nn_point;
	Matrix D_distance_vec;
	MatrixInt D_indices_vec;
	
	//TODO: implement
	//~ int num_points = readNumPointsFromFile(filename);
	int num_points = 50000000;
	
	generateHostMatrix(V,num_points,3);
	generateHostMatrix(indices_vec,k,1);
	int* H_ResultVector = (int*)malloc(sizeof(int) );
	//~ int* H_ResultVector;
	//~ cudaMallocManaged(&H_ResultVector, sizeof(int) );
	
	//~ printf("%d\n",*H_ResultVector);
	
	//TODO:implement
	//~ loadPointsFromFile(filename,V);
	fillMatrixWithRandomFloats(V);
	
	
	generateDeviceMatrices(D_V,D_nn_point,D_distance_vec,D_indices_vec,V.width,3,k);
	int* D_ResultVector;
	cudaMalloc((void**)&D_ResultVector, sizeof(int) );
	
	
	
	
	copyToDeviceMatrix(V,D_V);
	float epsilon = 0.5;
	
	int num_search_points = 1000;
	//~ int num_search_points = num_points;
	float progress = 0.0;
	int barWidth = 70;
	
	int pos = barWidth * progress;
	float last_epsilon=epsilon;
	bool DEBUG = false;
	clock_t knnstart, knnend;
	knnstart = clock();
	for(int i=0; i < num_search_points; i++)
	{
		
		
		clock_t calcstart, calcend;
		if(DEBUG){
			printf("\n");
			calcstart = clock();
		}
		
		epsilon = kNNGpu(D_V,D_nn_point,D_distance_vec,D_indices_vec,H_ResultVector,D_ResultVector,
				V.width,k,i,epsilon,DEBUG);
		copyToHostMatrix(D_indices_vec,indices_vec);
		
		if(DEBUG){
			calcend = clock();
		
			printf("knn GPU %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);
		}
	
		
		//printMatrixInt(indices_vec);
		
		if(!DEBUG){
			progress = ((i+1) * 100)/(num_search_points) ;
			std::cout << "Progress: " << progress << " % epsilon diff: " << epsilon-last_epsilon <<"\r";
			std::cout.flush();
		}
		
		last_epsilon=epsilon;
		
	}
	std::cout << std::endl;
	
	knnend = clock();
	
	printf("finished knn search: %d datapoints, searched_points: %d , k = %d\n",num_points,num_search_points,k);
	printf("total knn %f milliseconds\n",(float)(knnend-knnstart)*1000.0 / CLOCKS_PER_SEC);
		
	
	
	freeCudaMatrices(D_V,D_nn_point,D_distance_vec,D_indices_vec);
	cudaFree(D_ResultVector);
	free(V.elements);
	free(indices_vec.elements);
	free(H_ResultVector);
	
}



