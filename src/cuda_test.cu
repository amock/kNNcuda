#include <time.h>
#include <stdio.h>
#include <iostream>


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// Host global variables
// Device Information
int m_shared_memory_size;
int m_mps;
int m_threads_per_mp;
int m_threads_per_block;
int* m_size_thread_block;
int* m_size_grid;
unsigned long long m_device_global_memory;

//Device global variables
extern __shared__ float knn[];


__device__ float sumShared(unsigned int mid, int k)
{

    float res = 0.0;

    for( int i = 0; i < k; i++ )
    {
        res += knn[ mid + i ];
    }

    return res;
}

__device__ void fillShared( unsigned int mid, int k )
{


    for( unsigned int i = 0; i < k; i++ )
    {

        //printf("%u\n", i);
        knn[ mid + i ] = i ;
        //if(mid < 10){
        //    printf("i: %u, k: %d, mid: %u, value: %f \n", i, k, mid, knn[mid + i]);
        //}

    }
}

__global__ void testKernel( float* numbers, int n, int k) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mid = threadIdx.x * k;


    if(tid < n)
    {

        //knn[ mid ] = numbers[ tid ];
        // fill shared memory
        fillShared(mid, k);

    }
}

void fillRand(float* numbers, int n)
{
    srand (time(NULL));
    for( int i=0; i<n; numbers[i++] = rand() % 10 );
}

void fillSeq(float* numbers, int n)
{
    srand (time(NULL));
    for( int i=0; i<n; numbers[i++] = i );
}

void print(float* numbers, int n)
{
    for( int i=0; i<n; i++ )
    {
        std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;
}

void calculateBlocksThreads(int n, int elements, int element_size, int max_mem_shared, int max_threads_per_block,
                            int& out_blocks_per_grid, int& out_threads_per_block, int& needed_shared_memory)
{
    int mem_per_thread_needed = elements * element_size;

    out_threads_per_block = max_mem_shared / mem_per_thread_needed;

    if( out_threads_per_block > max_threads_per_block )
    {
        out_threads_per_block = max_threads_per_block;
    }

    out_blocks_per_grid = ( n + out_threads_per_block - 1 ) / out_threads_per_block;

    needed_shared_memory = out_threads_per_block * element_size * elements;
}

void getCudaInformation()
{

        m_mps = 0;
        m_threads_per_mp = 0;
        m_threads_per_block = 0;
        m_size_thread_block = new int(3);
        m_size_grid = new int(3);
        m_device_global_memory = 0;


        cudaSetDevice(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        m_shared_memory_size = deviceProp.sharedMemPerBlock;
        m_mps = deviceProp.multiProcessorCount;
        m_threads_per_mp = deviceProp.maxThreadsPerMultiProcessor;
        m_threads_per_block = deviceProp.maxThreadsPerBlock;
        m_size_thread_block[0] = deviceProp.maxThreadsDim[0];
        m_size_thread_block[1] = deviceProp.maxThreadsDim[1];
        m_size_thread_block[2] = deviceProp.maxThreadsDim[2];
        m_size_grid[0] = deviceProp.maxGridSize[0];
        m_size_grid[1] = deviceProp.maxGridSize[1];
        m_size_grid[2] = deviceProp.maxGridSize[2];
        m_device_global_memory = (unsigned long long) deviceProp.totalGlobalMem;

}

int main( int argc, const char** argv )
{
    getCudaInformation();
    int N = 10;
    if(argc > 1) {
        N = atoi(argv[1]);
    }

    int k = 50;
    if(argc > 2){
        k = atoi(argv[2]);
    }


    float* numbers = (float*)malloc( N * sizeof(float) );
    float* dev_numbers;

    HANDLE_ERROR(cudaMalloc( &dev_numbers, N * sizeof(float) ));
    //fillRand( numbers, N );
    fillSeq( numbers, N );

    //print(numbers, N);

    HANDLE_ERROR(cudaMemcpy(dev_numbers, numbers, N * sizeof(float), cudaMemcpyHostToDevice ));

    int blocks_per_grid, threads_per_block, needed_shared_memory;

    calculateBlocksThreads(N, k, sizeof(float), m_shared_memory_size, m_threads_per_block,
                           blocks_per_grid, threads_per_block, needed_shared_memory);


    printf("Blocks in use: %d\n", blocks_per_grid);
    printf("Threads per block: %d\n", threads_per_block);
    printf("Memory in use each block: %d/%d\n", needed_shared_memory, m_shared_memory_size);

    // <<< blocks_per_grid, threads_per_block, shared memory size >>>
    testKernel<<< blocks_per_grid, threads_per_block, needed_shared_memory >>>(dev_numbers, N, k);

    cudaMemcpy(dev_numbers, numbers, N * sizeof(float), cudaMemcpyDeviceToHost );

    //print(numbers, N);

    cudaFree(dev_numbers);
    free(numbers);

}
