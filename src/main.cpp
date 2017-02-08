/**
 * main.cpp
 *
 * @author Alexander Mock
 * @author Matthias Greshake
 */

#include "../include/calcNormals.h"

int m_mps, m_cuda_cores_per_mp, m_threads_per_mp, m_threads_per_block;
int* m_size_block = new int(3);
int* m_size_grid = new int(3);
unsigned long long m_device_global_memory;

void getDeviceInformation(int &mps, int &cuda_cores_per_mp, int &threads_per_mp, int &threads_per_block, int* size_block, int* size_grid, unsigned long long &device_global_memory) {
	cudaDeviceProp deviceProp;

	cudaSetDevice(0);
	cudaGetDeviceProperties(&deviceProp, 0);

	mps = deviceProp.multiProcessorCount;
	cuda_cores_per_mp = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	threads_per_mp = deviceProp.maxThreadsPerMultiProcessor;
	threads_per_block = deviceProp.maxThreadsPerBlock;
	size_block[0] = deviceProp.maxThreadsDim[0];
	size_block[1] = deviceProp.maxThreadsDim[1];
	size_block[2] = deviceProp.maxThreadsDim[2];
	size_grid[0] = deviceProp.maxGridSize[0];
	size_grid[1] = deviceProp.maxGridSize[1];
	size_grid[2] = deviceProp.maxGridSize[2];
	device_global_memory = (unsigned long long)deviceProp.totalGlobalMem;
}

int main(int argc, char** argv) {
        const char* file;
        const char* dest = "normals.ply";

	getDeviceInformation(m_mps, m_cuda_cores_per_mp, m_threads_per_mp, m_threads_per_block, m_size_block, m_size_grid, m_device_global_memory);

	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "--info") == 0) {
			cout << "Device Information:" << endl;
			cout << "Multi Processors: " << m_mps << endl;
			cout << "CUDA Cores per Multi Processor: " << m_cuda_cores_per_mp << endl;
			cout << "Threads per Multi Processor: " << m_threads_per_mp << endl;
			cout << "Threads per Block: " << m_threads_per_block << endl;
			cout << "Thread Dimension: " << m_size_block[0] << ", " << m_size_block[1] << ", " << m_size_block[2] << endl;
			cout << "Grid Size: " << m_size_grid[0] << ", " << m_size_grid[1] << ", " << m_size_grid[2] << endl;
			cout << "Device Global Memory: " << m_device_global_memory << endl;
			cout << endl;
		}
                if (strcmp(argv[i], "--file") == 0) {
                        file = argv[i+1];
                }
	}

        int seed = 12345;
	srand(seed);

        int k = 50;
        int numPoints = 100000;
        kNearestNeighborSearch(k, numPoints, file, dest);

        cout << endl << "Terminated" << endl;
}
