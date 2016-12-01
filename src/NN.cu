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

int m_mps = 0;
int m_cuda_cores_per_mp = 0;
int m_threads_per_mp = 0;
int m_threads_per_block = 0;
int* m_size_thread_block = new int(3);
int* m_size_grid = new int(3);
unsigned long long m_device_global_memory = 0;


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
		*(m.elements + i ) = ((float)rand()/(float)(RAND_MAX)) * 1000.0 ;
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


__global__ void DistanceKernel(const Matrix A, const Matrix B, Matrix dest);



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

	clock_t calcstart, calcend;
	calcstart = clock();

	// Invoke kernel
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (A.width +threadsPerBlock-1)/threadsPerBlock;

	DistanceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

	calcend = clock();
	printf("Distance calculation %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);

	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size,
	       cudaMemcpyDeviceToHost);


	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
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
	int A_size = A.width*A.height;
	//~ int threadsPerBlock = m_threads_per_block;
	//~ int blocksPerGrid = (A.width +threadsPerBlock-1)/threadsPerBlock;
	//~ MergeSort<<<1,A_size>>>(d_A,d_B);
	
	//~ int A_size = A.width*A.height;
	//MergeSort<<<1, A_size, sizeof(float)*A_size*2>>>(d_A,d_B);
	//~ int sort_iterations = log(A_size)/log(2);
	//~ std::cout << sort_iterations << std::endl;
	//~ for(int i=0;i<sort_iterations;i++)
	//~ {
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

//distance function without transformation
__global__ void DistanceKernel(const Matrix points,const Matrix s_point, Matrix dest)
{
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	int num_points = points.width;
	if(tid < num_points)
	{
		dest.elements[tid] = (points.elements[tid + 0 * points.width] - s_point.elements[0]) * (points.elements[tid + 0 * points.width] - s_point.elements[0]) 
								+ (points.elements[tid + 1 * points.width] - s_point.elements[1]) * (points.elements[tid + 1 * points.width] - s_point.elements[1])
								+ (points.elements[tid + 2 * points.width] - s_point.elements[2]) * (points.elements[tid + 2 * points.width] - s_point.elements[2]) ; 
	}
}

__device__ inline void Merge2(float* a, int i1, int j1, int i2, int j2,int limit=-1){
	
	
	
	float* temp = (float*) malloc((j2-i1+1) * sizeof(float));  //array used for merging
    int i,j,k;
    i=i1;    //beginning of the first list
    j=i2;    //beginning of the second list
    k=0;
    
    int counter = 0;
    while(i<=j1 && j<=j2 )    //while elements in both lists
    {
		counter ++;
        if(a[i]<a[j])
            temp[k++]=a[i++];
        else
            temp[k++]=a[j++];
    }
    
    while(i<=j1 )    //copy remaining elements of the first list
        temp[k++]=a[i++];
        
    while(j<=j2 )    //copy remaining elements of the second list
        temp[k++]=a[j++];
        
    //Transfer elements from temp[] back to a[]
    for(i=i1,j=0;i<=j2 ;i++,j++)
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


void naturalMergeSort(Matrix& m, int limit=-1){
	int m_elements = m.width*m.height;
	
	int slide_buffer_size = int(m_elements-0.5);
	int* slide_buffer = (int*) malloc(slide_buffer_size * sizeof(int));

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
	
	free(slide_buffer);
}

int main(int argc, char** argv)
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
	Matrix V;
	V.height = 4;
	V.width = 10;
	std::cout << "points " << V.width << std::endl; 
	V.stride = V.width;
	mallocMatrix(V);
	fillHomogenMatrixWithRandomFloats(V);
	printMatrix(V);
	
	
	Matrix last_point_V;
	last_point_V.height = 4;
	last_point_V.width = 1;
	last_point_V.stride = last_point_V.width;
	mallocMatrix(last_point_V);
	getColVecOfMatrix(V,V.width-1,last_point_V);
	// Matrix initialized
	//printMatrix(last_point_V);
	
	//point index for searching
	int index = 1;
	
	//point for searchings
	Matrix nn_point;
	nn_point.height = 4;
	nn_point.width = 1;
	nn_point.stride = nn_point.width;
	mallocMatrix(nn_point);
	getColVecOfMatrix(V,index,nn_point);
	std::cout << "pic point " << index << std::endl;
	//printMatrix(nn_point);
	printMatrix(nn_point);
	//neue variante ohne transformation
	Matrix distance_vec;
	distance_vec.height = 1;
	distance_vec.width = V.width;
	distance_vec.stride = distance_vec.width;
	mallocMatrix(distance_vec);
	
	Distances(V,nn_point,distance_vec);
	printMatrix(distance_vec);
	
	//transformation with point
	Matrix T;
	T.height = 4;
	T.width = 4;
	T.stride = T.width;
	mallocMatrix(T);
	fill3DTranslationMatrixRightCol(T,-nn_point.elements[0],-nn_point.elements[1],-nn_point.elements[2]);
	// Transformation matrix initialized
	
	
	Matrix V1;
	V1.height = V.height;
	V1.width = V.width;
	V1.stride = V1.width;
	mallocMatrix(V1);
	
	
	//printMatrix(T);
	//printMatrix(V);
	
	// Transformation
	// auf großen punktmengen geht das nicht mehr
	MatMul(T,V,V1);

	Matrix last_point_V1;
	last_point_V1.height = 4;
	last_point_V1.width = 1;
	last_point_V1.stride = last_point_V1.width;
	mallocMatrix(last_point_V1);
	getColVecOfMatrix(V1,V1.width-1,last_point_V1);
	// Matrix initialized
	printMatrix(last_point_V1);
	
	//self multiplication
	//Matrix Vtransposed;
	//Vtransposed.height = V1.width;
	//Vtransposed.width = V1.height;
	//Vtransposed.stride = Vtransposed.width;
	//mallocMatrix(Vtransposed);
	//transposeMatrix(V1,Vtransposed);
	
	Matrix V2;
	V2.height = 1;
	V2.width = V1.width;
	V2.stride = V2.width;
	mallocMatrix(V2);

	// Snow Shovel
	SelfScalar(V1, V2);


	Matrix last_point;
	last_point.height = 4;
	last_point.width = 1;
	last_point.stride = last_point.width;
	mallocMatrix(last_point);
	getColVecOfMatrix(V1,V1.width-1,last_point);
	

	printMatrix(last_point);
	std::cout << "scalar: " << V2.elements[V2.width*V2.height-1] << std::endl;
	
	
	
	//printMatrix(V1);
	//printMatrix(V2);	


	Matrix sortedVector;
	sortedVector.height = 1;
	sortedVector.width = V2.width;
	sortedVector.stride = sortedVector.width;
	mallocMatrix(sortedVector);

	printMatrix(V2);
	
	Sort(V2,sortedVector);
	
	//std::cout << "Sorted Last Point: " << sortedVector.elements[sortedVector.width*sortedVector.height-1] << std::endl;
	//printMatrix(V2);
	printMatrix(sortedVector);

	//Sort(sortedVector,sortedVector);
	//Sort(sortedVector,sortedVector);
	//Sort(sortedVector,sortedVector);
	
	//printMatrix(sortedVector);
	//Matrix Weights;
	//Weights.height = Vtransposed.height;
	//Weights.width = V1.width;
	//Weights.stride = Vtransposed.width;
	//mallocMatrix(Weights);
	
	
	//printMatrix(Vtransposed);
	//printMatrix(V1);
	
	//MatMul(Vtransposed,V1,Weights);
	//std::cout << "last Weight" << Weights.elements[0] << std::endl;
	//std::cout << "last Weight " << Weights.elements[Weights.width*Weights.height-1] << std::endl;
	
	//diagonal is distance
	//printMatrix(Weights);
	
	
	
	//printDeviceInformation();
	
	
	
	
	
	//~ Matrix unsortedVector;
	//~ unsortedVector.height = 1;
	//~ unsortedVector.width = 12;
	//~ unsortedVector.stride = unsortedVector.width;
	//~ mallocMatrix(unsortedVector);
	//~ fillMatrixWithRandomFloats(unsortedVector);
	//~ printMatrix(unsortedVector);
	
	//~ naturalMergeSort(V2,50);

	

	//~ free(unsortedVector.elements);
	free(sortedVector.elements);
	free(V.elements);
	//free(Vtransposed.elements);
	free(T.elements);
	free(V1.elements);
	free(V2.elements);
	free(last_point.elements);
	free(last_point_V.elements);
	free(last_point_V1.elements);
	free(nn_point.elements);
	
	return 0;
}



