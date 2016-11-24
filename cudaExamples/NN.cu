#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

// Thread block size
#define BLOCK_SIZE 1

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;


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
			m1.elements[i*m1.width+j] = int(val);
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


void firstTry(int argc, char** argv){
	clock_t prgstart, prgend;
	prgstart = clock();
	
	int seed = 1479731956;
	//int seed = time(NULL);
	printf("%d\n",seed);
    srand(seed);
    
    // ergebnis matrix: firstMatrixHeight x secondMatrixWidth
    int firstMatrixHeight = 20;
    int secondMatrixWidth = 20;
    int firstSecondLength = 300;
    if(argc > 1){
		firstSecondLength = atoi(argv[1]);
		
	}
	
	printf("%dx%d * %dx%d\n",firstSecondLength,firstMatrixHeight,secondMatrixWidth,firstSecondLength);
	
	Matrix hMatA;
	hMatA.height = firstMatrixHeight;
	hMatA.width = firstSecondLength;
	hMatA.stride = hMatA.width;
	mallocMatrix(hMatA);
	fillMatrixWithRandomFloats(hMatA);
	//printMatrix(hMatA);
	
	Matrix hMatB;
	hMatB.height = firstSecondLength;
	hMatB.width = secondMatrixWidth;
	hMatB.stride = hMatB.width;
	mallocMatrix(hMatB);
	fillMatrixWithRandomFloats(hMatB);
	//printMatrix(hMatB);
	
	//IMPORTANT change height and widht
	Matrix hMatC;
	hMatC.height = firstMatrixHeight;
	hMatC.width = secondMatrixWidth;
	hMatC.stride = hMatC.width;
	mallocMatrix(hMatC);
	
	
	printf("MULTIPLY!\n");
	MatMul( hMatA,hMatB, hMatC);
	
	
	//printMatrix(hMatC);
	
	free(hMatA.elements);
	free(hMatB.elements);
	free(hMatC.elements);
	
	prgend=clock();
	printf("Laufzeit insgesamt %f seconds\n",(float)(prgend-prgstart) / CLOCKS_PER_SEC);
	
}


int main(int argc, char** argv)
{
	
	//plan: (T * v_points) * (T * v_points)^T 
	int seed = 1479731956;
	//int seed = time(NULL);
	printf("%d\n",seed);
    srand(seed);
    
	//point vector
	Matrix V;
	V.height = 4;
	V.width = 5;
	V.stride = V.width;
	mallocMatrix(V);
	fillHomogenMatrixWithRandomFloats(V);
	
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
	printMatrix(nn_point);
	
	//transformation with point
	Matrix T;
	T.height = 4;
	T.width = 4;
	T.stride = T.width;
	mallocMatrix(T);
	fill3DTranslationMatrixRightCol(T,-nn_point.elements[0],-nn_point.elements[1],-nn_point.elements[2]);
	
	
	Matrix V1;
	V1.height = V.height;
	V1.width = V.width;
	V1.stride = V1.width;
	mallocMatrix(V1);
	
	
	printMatrix(T);
	printMatrix(V);
	
	MatMul(T,V,V1);
	
	//self multiplication
	Matrix Vtransposed;
	Vtransposed.height = V1.width;
	Vtransposed.width = V1.height;
	Vtransposed.stride = Vtransposed.width;
	mallocMatrix(Vtransposed);
	transposeMatrix(V1,Vtransposed);
	
	
	
	Matrix Weights;
	Weights.height = Vtransposed.height;
	Weights.width = V1.width;
	Weights.stride = Vtransposed.width;
	mallocMatrix(Weights);
	
	
	printMatrix(Vtransposed);
	printMatrix(V1);
	
	MatMul(Vtransposed,V1,Weights);
	
	//diagonal is distance
	printMatrix(Weights);
	
	free(V.elements);
	free(Vtransposed.elements);
	free(T.elements);
	free(V1.elements);
	free(nn_point.elements);
	
	return 0;
}



int main2(int argc, char** argv){
	std::cout << "Hello World" << std::endl;
	std::vector<float> point_vec;
	point_vec.push_back(0.1);
	for(std::vector<float>::iterator it = point_vec.begin(); it != point_vec.end(); ++it)
	{
		std::cout << *it << std::endl;
	}
	printf("hello world\n");
	return 0;
}

