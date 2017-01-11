#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <float.h>
#include "../include/helper_cuda.h"

int m_mps = 0;
int m_cuda_cores_per_mp = 0;
int m_threads_per_mp = 0;
int m_threads_per_block = 0;
int* m_size_thread_block = new int(3);
int* m_size_grid = new int(3);
unsigned long long m_device_global_memory = 0;

struct betterKdTree {
	//node and leef
	float value;
	struct betterKdTree *left, *right;
};

struct Matrix {
    int width;
    int height;
    int stride; 
    float* elements;
};

struct MatrixInt {
	int width;
    int height;
    int stride; 
    int* elements;
};

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




void mallocMatrix(Matrix& m){
	m.elements = (float*)malloc(m.width * m.height * sizeof(float));
}


void generateHostMatrix(Matrix& m, int width, int height){
	
	
	m.height = height;
	m.width = width;
	m.stride = m.width;
	m.elements = (float*)malloc(m.width * m.height * sizeof(float) );
	
}

void generateDeviceMatrix(Matrix& D_m, int width, int height){
	
    D_m.width = D_m.stride = width;
    D_m.height = height;
    size_t size = D_m.width * D_m.height * sizeof(float);
    cudaMalloc(&D_m.elements, size);
    
}

void copyToDeviceMatrix(Matrix& m, Matrix& D_m){
	
	size_t size = m.width * m.height * sizeof(float);
    cudaMemcpy(D_m.elements, m.elements, size, cudaMemcpyHostToDevice);

}

void copyToHostMatrix(Matrix& D_m, Matrix& m){
	
	size_t size = m.width * m.height * sizeof(float);
	printf("size: %d\n", (int)size);
	cudaMemcpy(m.elements, D_m.elements, size, cudaMemcpyDeviceToHost);
	
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

void fillMatrixWithSequence(Matrix& m){
	for(int i=0;i<m.width*m.height;i++){
		m.elements[i] = i;
	}
}  

void copyRowToMatrix(Matrix& in, int row, Matrix& out){
	for(int i = 0; i<in.width*(row+1) && i<out.width; i++){
		out.elements[i] = in.elements[i+in.width*(row)];
	}
}

void copyVectorInterval(Matrix& in,int start, int end, Matrix& out){
	for(int i=0; i < (end-start); i++){
		out.elements[i] = in.elements[i+start];
	}
}




__global__ void KNNKernel(const Matrix D_V, const Matrix D_kd_tree, Matrix D_Result_Normals, int k=50);



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


void mergeHostWithIndices(float* a, float* b, int i1, int j1, int i2, int j2,int limit=-1){
	int limit_end = limit;
	
	
	float* temp = (float*) malloc((j2-i1+1) * sizeof(float));  //array used for merging
    int* temp_indices = (int*) malloc((j2-i1+1) * sizeof(int));  //array used for merging
    
    
    int i,j,k;
    i=i1;    //beginning of the first list
    j=i2;    //beginning of the second list
    k=0;
    
    int counter = 0;
    while(i<=j1 && j<=j2 && limit!=0)    //while elements in both lists
    {
		counter ++;
		limit--;
        if(a[i]<a[j]){
			temp_indices[k] = b[i]; 
            temp[k++]=a[i++];
            
        }else{
			temp_indices[k] = b[j];
            temp[k++]=a[j++];
		}
    }
    
    while(i<=j1 && limit!=0) {   //copy remaining elements of the first list
		temp_indices[k] = b[i]; 
        temp[k++]=a[i++];
	}
        
    while(j<=j2 && limit!=0) {   //copy remaining elements of the second list
        temp_indices[k] = b[j]; 
        temp[k++]=a[j++];
	}
        
    //Transfer elements from temp[] back to a[]
    for(i=i1,j=0;i<=j2 && limit_end!=0 ;i++,j++,limit_end--)
	{
		b[i] = temp_indices[j];
		if(b[i] < 0){
			printf("THERE IS SOMETHING WRONG\n");
		}
        a[i] = temp[j];
    }   
    free(temp_indices);
    free(temp);
}


void naturalMergeSort(Matrix& in, int dim, Matrix& indices,  Matrix& m, int limit=-1){
	
	copyRowToMatrix(in, dim, m);
	
	//~ printf("copy row of Mat for dim %d\n", dim);
	//~ printMatrix(m,true);
	
	int m_elements = m.width * m.height;
	
	int slide_buffer_size = int(m_elements-0.5);
	int* slide_buffer = (int*) malloc(slide_buffer_size * sizeof(int));

	clock_t calcstart, calcend;
	calcstart = clock();

	//create RUNS
	int num_slides = 1;
	slide_buffer[0] = 0;
	for(int i=1; i < slide_buffer_size+1; i++) {
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
		//~ std::cout << count+1 <<" Iteration: You can use " << int(num_slides/2) << " Threads" << std::endl;
		
		int i;
		
		for(i=2;i<int(num_slides+1);i+=2)
		{
				
			mergeHostWithIndices(m.elements, indices.elements ,slide_buffer[i-2], slide_buffer[i-1]-1, slide_buffer[i-1], slide_buffer[i]-1,current_limit);
			
			
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

void sortByDim(Matrix& V, int dim, Matrix& indices, Matrix& values){
	naturalMergeSort(V, dim, indices, values);
}

void calculateMedian(Matrix& V, Matrix& indices, int current_dim, float& median_index, float& median_value, int num_medians) {
	
	median_index = indices.elements[indices.width/2+1];
	median_value = V.elements[V.width*current_dim+(int)median_index];
}



void splitMatrix(Matrix& I, Matrix& I_L, Matrix& I_R){
	
	int i=0;
	for(; i<I_L.width; i++){
		I_L.elements[i] = I.elements[i];
	}
	int j=0;
	for(; i<I.width && j<I_R.width; i++, j++){
		I_R.elements[j] = I.elements[i];
	}
	
}

void splitMatrixWithValue(Matrix& V, Matrix& I, Matrix& I_L, Matrix& I_R, int current_dim, float value){
	int i_l = 0;
	int i_r = 0;
	
	//~ printMatrix(V);
	//~ printMatrix(I);
	
	//~ printf("split by value: %f\n",value);
	//~ printf("splitting array (%d) to (%d, %d) with value %f\n",I.width,I_L.width,I_R.width,value);
	for(int i=0; i<I.width; i++){
		float current_value = V.elements[current_dim*V.width + static_cast<int>(I.elements[i]+0.5) ];
		//~ printf("curr val: %f\n", current_value);
		if(current_value <= value && I_L.width > i_l ){
			//~ printf("add to left: %f with value %f\n", I.elements[i], current_value);
			I_L.elements[i_l++] = I.elements[i];
		}else if(current_value >= value && I_R.width > i_r){
			//~ printf("add to right: %f with value %f\n", I.elements[i], current_value);
			I_R.elements[i_r++] = I.elements[i];
		}else {
			if(i_r<I_R.width){
				I_R.elements[i_r++] = I.elements[i];
			}else if(i_l<I_L.width){
				I_L.elements[i_l++] = I.elements[i];
			}
		}
	}
	
	if(i_l != I_L.width){
		printf("WARNING left %d != %d\n",i_l,I_L.width);
	}
	
	if(i_r != I_R.width){
		printf("WARNING right %d != %d\n",i_r,I_R.width);
	}
		
}

void generateKdTreeRecursive(Matrix& V, Matrix* sorted_indices, int current_dim, int max_dim, Matrix& kd_tree, int size, int max_tree_depth, int position){
	
	int left = position*2+1;
	int right = position*2+2;
	
	if(right > size-1 || left > size-1){
		//
		
		kd_tree.elements[position] = sorted_indices[current_dim].elements[0];
		//~ printf("leaf! pos: %d val: %f\n",position, kd_tree.elements[position]);
		
	}else{
		/// split sorted_indices
		int indices_size = sorted_indices[current_dim].width;
		
		// calculate left balanced sizes
		int next_pot = static_cast<int>(log2f(indices_size-1));
		int right_size = pow(2,next_pot-1);
		int left_size = indices_size - right_size;
		int val_next_pot = pow(2,next_pot);
		if( left_size > val_next_pot ){
			right_size += left_size - val_next_pot;
			left_size = val_next_pot;
		}
		
		float split_value = (V.elements[current_dim*V.width+static_cast<int>(sorted_indices[current_dim].elements[left_size-1]) ] + V.elements[current_dim*V.width+static_cast<int>(sorted_indices[current_dim].elements[left_size] ) ] ) /2.0;
		
		kd_tree.elements[position] = split_value;
		
		struct Matrix sorted_indices_left[max_dim];
		struct Matrix sorted_indices_right[max_dim];
		
		// alloc new memory
		for(int i=0; i<max_dim; i++){
			//memory corruption when malloc 
			
			sorted_indices_left[i].width = left_size;
			sorted_indices_left[i].height = 1;
			sorted_indices_left[i].elements = (float*)malloc( (left_size+1) *sizeof(float) );
			
			sorted_indices_right[i].width = right_size;
			sorted_indices_right[i].height = 1;
			sorted_indices_right[i].elements = (float*)malloc( (right_size+1) * sizeof(float) );
			
			if(i==current_dim){
				splitMatrix(sorted_indices[i], sorted_indices_left[i], sorted_indices_right[i]);
			}else{
				splitMatrixWithValue(V, sorted_indices[i], sorted_indices_left[i], sorted_indices_right[i], current_dim, split_value);
			}
			
		}
		
		generateKdTreeRecursive(V, sorted_indices_left, (current_dim+1)%max_dim, max_dim, kd_tree, size, max_tree_depth, left);
		generateKdTreeRecursive(V, sorted_indices_right, (current_dim+1)%max_dim, max_dim, kd_tree, size, max_tree_depth, right);		
	
		
		// alloc new memory
		for(int i=0; i<max_dim; i++){
			free(sorted_indices_left[i].elements);
			free(sorted_indices_right[i].elements);
		}
	}
	
	
}

void generateKdTreeArray(Matrix& V, Matrix* sorted_indices, int max_dim, Matrix& kd_tree, int& size, int& max_tree_depth){
	
	
	max_tree_depth = static_cast<int>(log2f(V.width-1)+2.0) ;
	int max_leaf_size = static_cast<int>(pow(2,max_tree_depth) );
	
	if(V.width == 1){
		max_tree_depth = 1;
	}
	printf("tree depth: %d\n",max_tree_depth);
	
	size = V.width*2-1;
	
	printf("calulated kd-tree size: %d\n",size);
	generateHostMatrix(kd_tree, size, 1);
	
	//start real generate
	generateKdTreeRecursive(V, sorted_indices, 0 ,max_dim, kd_tree, size, max_tree_depth, 0);
	
	
}

// Get a matrix element
__device__ int GetKdTreePosition(const Matrix& D_kd_tree, float x, float y, float z)
{
	int pos = 0;
	int current_dim = 0;
	while(pos*2+1 < D_kd_tree.width){
		if(current_dim == 0){
			if(x <= D_kd_tree.elements[pos] ){
				pos = pos*2+1;
			}else{
				pos = pos*2+2;
			}
			
			current_dim += 1;
		}else if(current_dim == 1){
			if(y <= D_kd_tree.elements[pos] ){
				pos = pos*2+1;
			}else{
				pos = pos*2+2;
			}
			
			current_dim +=1;
		}else{
			if(z <= D_kd_tree.elements[pos] ){
				pos = pos*2+1;
			}else{
				pos = pos*2+2;
			}
			
			current_dim = 0;
		}
		
	}
	
    return pos;
}

__device__ float SearchQueryPoint(const Matrix& D_kd_tree, float x, float y, float z){
	return D_kd_tree.elements[GetKdTreePosition(D_kd_tree, x, y, z)];
}

__device__ void calculateNormalFromSubtree(const Matrix& D_V, const Matrix& D_kd_tree, int sub_tree_index, float& x,float& y,float& z){
	x = 1.0;
	y = 1.0;
	z = 1.0;
} 

//distance function without transformation
__global__ void KNNKernel(const Matrix D_V, const Matrix D_kd_tree, Matrix D_Result_Normals, int k)
{
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(tid < D_V.width){
		
		int pos = GetKdTreePosition(D_kd_tree, D_V.elements[tid], D_V.elements[D_V.width + tid], D_V.elements[D_V.width*2 + tid] );
		
		float val = SearchQueryPoint(D_kd_tree, D_V.elements[tid], D_V.elements[D_V.width + tid], D_V.elements[D_V.width*2 + tid] );
		
		float result_x = 0.0;
		float result_y = 0.0;
		float result_z = 0.0;
		calculateNormalFromSubtree(D_V, D_kd_tree, pos, result_x, result_y, result_z);
		
		//~ D_Result_Normals.elements[tid] = result_x;
		//~ D_Result_Normals.elements[D_Result_Normals.width + tid] = result_y;
		//~ D_Result_Normals.elements[D_Result_Normals.width*2 +tid] = result_z;
		printf("result_x %f\n",D_Result_Normals.elements[tid]);
		D_Result_Normals.elements[tid] = 1.0;
		printf("result_x %f\n",D_Result_Normals.elements[tid]);
		printf("query result: %f \n");
		
	}
	
}

void GPU_NN(Matrix& D_V, Matrix& D_kd_tree, Matrix& D_Result_Normals, Matrix& Result_Normals){
	printf("START\n");
	
	int threadsPerBlock = m_threads_per_block;
	int blocksPerGrid = (D_V.width + threadsPerBlock-1)/threadsPerBlock;

	KNNKernel<<<blocksPerGrid, threadsPerBlock>>>(D_V, D_kd_tree, D_Result_Normals);
	cudaDeviceSynchronize();
	
	size_t size = Result_Normals.width * Result_Normals.height * sizeof(float);
	printf("size: %d\n", (int)size);
	
	cudaMemcpy(Result_Normals.elements, D_Result_Normals.elements, size, cudaMemcpyDeviceToHost);
	
	printf("END\n");
}


int main(int argc, char** argv)
{
	getCudaInformation(m_mps, m_cuda_cores_per_mp, m_threads_per_mp, m_threads_per_block, m_size_thread_block, m_size_grid, m_device_global_memory);
	
	const char * filename  = "points.ply";
	
	//HOST STUFF
	int point_dim = 3;
	int num_points = 4;
	
	if(argc > 1){
		num_points = atoi(argv[1]);
	}
	
	int k=50;
	int dim_points = 3;
	
	Matrix V,Result_Normals;
	struct Matrix test;
	struct Matrix indices_sorted[point_dim];
	struct Matrix values_sorted[point_dim];
	
	generateHostMatrix(test, num_points, 1);
	generateHostMatrix( V, num_points, point_dim);
	generateHostMatrix( Result_Normals, num_points, point_dim);
	fillMatrixWithRandomFloats( V);
	fillMatrixWithRandomFloats( Result_Normals);
	
	
	
	for(int i=0; i < point_dim; i++)
	{
		printf("generate indices for dim %d\n",i);
		generateHostMatrix(indices_sorted[i], V.width, 1);
		
		printf("generate values for dim %d\n",i);
		generateHostMatrix(values_sorted[i], V.width,1);
		fillMatrixWithSequence(indices_sorted[i]);
		
		sortByDim( V, i, indices_sorted[i] , values_sorted[i]);
	}
	//~ printMatrix(V);
	
	printf("Start generating kd-tree array based\n");
	//sorted indices + values
	//do some stuff
	Matrix kd_tree;
	int size = 0;
	int max_tree_depth = 0;
	
	generateKdTreeArray(V, indices_sorted, dim_points, kd_tree, size, max_tree_depth);
	
	//~ printMatrix(V);
	printf("End generating kd-tree array based\n");
	
	
	//push values to device
	//push kd_tree to device
	//DEVICE STUFF
	Matrix D_V, D_kd_tree, D_Result_Normals;
	generateDeviceMatrix( D_V, V.width, V.height );
	generateDeviceMatrix( D_kd_tree, kd_tree.width, kd_tree.height);
	generateDeviceMatrix( D_Result_Normals, Result_Normals.width, Result_Normals.height);
	
	//COPY STUFF
	copyToDeviceMatrix( V, D_V);
	copyToDeviceMatrix( kd_tree, D_kd_tree);
	copyToDeviceMatrix( Result_Normals, D_Result_Normals);
	
	
	//Cuda Kernels
	GPU_NN(D_V, D_kd_tree, D_Result_Normals, Result_Normals);
	
	
	
	printMatrix(Result_Normals);
	
	cudaFree(D_V.elements);
	cudaFree(D_kd_tree.elements);
	cudaFree(D_Result_Normals.elements);
	
	
	for(int i=0; i<point_dim;i++)
	{
		printf("free indices dim %d\n",i+1);
		free(indices_sorted[i].elements);
		
		printf("free values dim %d\n",i+1);
		free(values_sorted[i].elements);
	}
	
	free(V.elements);
	free(test.elements);
	free(Result_Normals.elements);
	printf("Free kd_tree array\n");
	free(kd_tree.elements);
}
