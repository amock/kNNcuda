#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "../include/helper_cuda.h"

// Thread block size
#define BLOCK_SIZE 4

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
		*(m.elements + i ) = ((float)rand()/(float)(RAND_MAX)) * 10.0 -5.0 ;
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
__global__ void SortKernel(Matrix m, int limit=-1);



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

void Sort(Matrix& A, int limit=-1)
{
	// Load A to device memory
	Matrix d_A;
	d_A.width = d_A.stride = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
	       cudaMemcpyHostToDevice);         


	clock_t calcstart, calcend;
	calcstart = clock();

	// Invoke kernel
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (A.width +threadsPerBlock-1)/threadsPerBlock;

	SortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A,limit);

	calcend=clock();
	printf("Sorted %f milliseconds\n",(float)(calcend-calcstart)*1000.0 / CLOCKS_PER_SEC);

	// Read C from device memory
	cudaMemcpy(A.elements, d_A.elements, size,
	       cudaMemcpyDeviceToHost);


	// Free device memory
	cudaFree(d_A.elements);
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

__device__ void merge(float* a, int i1, int j1, int i2, int j2,int limit=-1){
	
	int limit_end = limit;
	
	
	float* temp = (float*) malloc((j2-i1+1) * sizeof(float));  //array used for merging
    
	//float* temp;
	//size_t size = (j2-i1+1) * sizeof(float);
	//cudaMalloc(temp, size);





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
	//cudaFree(temp);
}

__global__ void SortKernel(Matrix m, int limit){
	//int limit = -1;
	
	
	int m_elements = m.width*m.height;
	
	int slide_buffer_size = int(m_elements-0.5);
	int* slide_buffer = (int*) malloc(slide_buffer_size * sizeof(int));
	//int* slide_buffer;	
	//size_t size = slide_buffer_size * sizeof(int);
	//cudaMalloc(slide_buffer, size);

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
		if(num_slides > 2) {
			current_limit = limit;
		}
		//std::cout << count+1 <<" Iteration: You can use " << int(num_slides/2) << " Threads" << std::endl;
		//printf("%d Runs\n",num_slides);
		
		//printf("%d Iteration: You can use %d Threads\n",count,int(num_slides/2));
		
		
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if(i>=2 && i%2==0 && i<int(num_slides+1) )
		{
			printf("Index %d \n",i);
			//parallelisierbar
			merge(m.elements, slide_buffer[i-2], slide_buffer[i-1]-1, slide_buffer[i-1], slide_buffer[i]-1,current_limit);
			__syncthreads();
			slide_buffer[i/2-1]= slide_buffer[i-2];
			slide_buffer[i/2]= slide_buffer[i];
		}

		if(num_slides%2 == 1){
			slide_buffer[(num_slides+1)/2] = slide_buffer[num_slides];
		}
		
		__syncthreads();
		printf("%d\n",num_slides);
		count ++;
		num_slides = int(num_slides/2.0+0.5);
		
	}
	
	free(slide_buffer);
	//cudaFree(slide_buffer);
}

void printDeviceInformation()
{
	int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
    
    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        //printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        //printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        //~ printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        char msg[256];
        SPRINTF(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        printf("%s", msg);

        printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        //printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        //printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        //printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

        //if (deviceProp.l2CacheSize)
        //{
        //    printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        //}

#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
        //int memoryClock;
        //getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        //printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        //int memBusWidth;
        //getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        //printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        //int L2CacheSize;
        //getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        //if (L2CacheSize)
        //{
        //    printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        //}

#endif
		//~ printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               //~ deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
               //~ deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        //~ printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               //~ deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        //~ printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
               //~ deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        //~ printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        //~ printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
        //~ printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        //~ printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        //~ printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        //~ printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        //~ printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        //~ printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        //~ printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
//~ #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        //~ printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
//~ #endif
        //~ printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        //~ printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

        //~ const char *sComputeMode[] =
        //~ {
            //~ "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            //~ "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            //~ "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            //~ "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            //~ "Unknown",
            //~ NULL
        //~ };
        //~ printf("  Compute Mode:\n");
        //~ printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }
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
	while(num_slides > 1){
		std::cout << count+1 <<" Iteration: You can use " << int(num_slides/2) << " Threads" << std::endl;
		int i;
		if(num_slides>2)
		{
			for(i=2;i<int(num_slides+1);i+=2)
			{
				//parallelisierbar
				mergeHost(m.elements, slide_buffer[i-2], slide_buffer[i-1]-1, slide_buffer[i-1], slide_buffer[i]-1);
				
				slide_buffer[i/2-1]= slide_buffer[i-2];
				slide_buffer[i/2]= slide_buffer[i];
			}
			
			if(num_slides%2 == 1){
				slide_buffer[i/2] = slide_buffer[num_slides];
			}
		}else{
			for(i=2;i<int(num_slides+1);i+=2)
			{
				//parallelisierbar
				mergeHost(m.elements, slide_buffer[i-2], slide_buffer[i-1]-1, slide_buffer[i-1], slide_buffer[i]-1,limit);
				
				slide_buffer[i/2-1]= slide_buffer[i-2];
				slide_buffer[i/2]= slide_buffer[i];
			}
			
			if(num_slides%2 == 1){
				slide_buffer[i/2] = slide_buffer[num_slides];
			}
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
	//int seed = 1479731956;
	int seed = time(NULL);
	printf("%d\n",seed);
    	srand(seed);
    
	//point vector ALLE PUNKTE
	Matrix V;
	V.height = 4;
	V.width = 100;
	std::cout << "points " << V.width << std::endl; 
	V.stride = V.width;
	mallocMatrix(V);
	fillHomogenMatrixWithRandomFloats(V);

	Matrix last_point_V;
	last_point_V.height = 4;
	last_point_V.width = 1;
	last_point_V.stride = last_point_V.width;
	mallocMatrix(last_point_V);
	getColVecOfMatrix(V,V.width-1,last_point_V);
	// Matrix initialized
	printMatrix(last_point_V);
	
	//point index for searching
	int index = 0;
	
	//point for searchings
	Matrix nn_point;
	nn_point.height = 4;
	nn_point.width = 1;
	nn_point.stride = nn_point.width;
	mallocMatrix(nn_point);
	
	getColVecOfMatrix(V,index,nn_point);
	std::cout << "pic point " << index << std::endl;
	//printMatrix(nn_point);
	
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
	
	
	
	
	
	Matrix unsortedVector;
	unsortedVector.height = 1;
	unsortedVector.width = 1000;
	unsortedVector.stride = unsortedVector.width;
	mallocMatrix(unsortedVector);
	fillMatrixWithRandomFloats(unsortedVector);
	//~ printMatrix(unsortedVector);
	
	Matrix sortedVector;
	sortedVector.height = 1;
	sortedVector.width = unsortedVector.width;
	sortedVector.stride = sortedVector.width;
	mallocMatrix(sortedVector);
	
	//naturalMergeSort(V2,50);

	Sort(V2,50);
	//~ std::cout << "Sorted: " << std::endl;
	printMatrix(V2);

	free(unsortedVector.elements);
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


