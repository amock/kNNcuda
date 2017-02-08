/**
 * kNN.cu
 *
 * @author Alexander Mock
 * @author Matthias Greshake
 */

#include "../include/calcNormals.h"


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

void evalEpsilon(Matrix &distVec, int* numNeighb, int* d_numNeighb, int k, float &epsilon, int tol=5) {
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
        if (*numNeighb > k + 1 + tol) {
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
        else break;
    } while (*numNeighb != k + 1);
}

void getNearestNeighbors(Matrix &distVec, Matrix &idxVec, float epsilon) {
    int threadsPerBlock = m_threads_per_block;
    int blocksPerGrid = (distVec.width + threadsPerBlock - 1) / threadsPerBlock;

    // Get nearest neighbors
    initIndex<<<1, 1>>>();
    getIndices<<<blocksPerGrid, threadsPerBlock>>>(distVec, idxVec, epsilon);
}

void getDirections(const Matrix &idxVec, const Matrix &pointMat, Matrix &neighbMat, int index) {
    for (int i = 0; i < neighbMat.height; i++) {
        for (int j = 0; j < neighbMat.width; j++)
            neighbMat.elements[i*neighbMat.width+j] = pointMat.elements[i*pointMat.width+(int)(idxVec.elements[j])] - pointMat.elements[i*pointMat.width+index];
    }
}

void calcNormals(const Matrix& neighbMat, Matrix& normalMat, int index, int max_iterations=5) {
    float min_dist = FLT_MAX;
    int iterations = 0;

    for (int i = 3; i < 3 * neighbMat.width; i += 3) {
        int j = (i + int(neighbMat.width / 3) * 3) % (neighbMat.width * 3);

        float x = neighbMat.elements[j+1] * neighbMat.elements[i+2] - neighbMat.elements[j+2] * neighbMat.elements[i+1];
        float y = neighbMat.elements[j+2] * neighbMat.elements[i+0] - neighbMat.elements[j+0] * neighbMat.elements[i+2];
        float z = neighbMat.elements[j+0] * neighbMat.elements[i+1] - neighbMat.elements[j+1] * neighbMat.elements[i+0];

        float norm = sqrtf(x * x + y * y + z * z);

        if (norm != 0.0) {
            x = 1.0 / norm * x;
            y = 1.0 / norm * y;
            z = 1.0 / norm * z;

            float cum_dist = 0.0;

            for (int j = 0; j < 3 * neighbMat.width; j += 3)
                cum_dist += abs(x * neighbMat.elements[j] + y * neighbMat.elements[j+1] + z * neighbMat.elements[j+2]);

            if(cum_dist < min_dist) {
                normalMat.elements[0*normalMat.width+index] = x;
                normalMat.elements[1*normalMat.width+index] = y;
                normalMat.elements[2*normalMat.width+index] = z;
            }
            else if (iterations < max_iterations) {
                iterations++;
            }
            else {
                return;
            }
        }
    }
}

int readVertexCb(p_ply_argument argument) {
    float ** ptr;
    ply_get_argument_user_data(argument, (void **) &ptr, NULL);
    **ptr = ply_get_argument_value(argument);
    (*ptr)++;
    return 1;
}

int readColorCb (p_ply_argument argument) {
    uint8_t ** color;
    ply_get_argument_user_data(argument, (void **) &color, NULL);
    **color = ply_get_argument_value(argument);
    (*color)++;
    return 1;
}

int readFaceCb(p_ply_argument argument) {
    unsigned int ** face;
    long int length, value_index;
    ply_get_argument_user_data(argument, (void **) &face, NULL);
    ply_get_argument_property(argument, NULL, &length, &value_index);
    if (value_index < 0) {
        /* We got info about amount of face vertices. */
        if (ply_get_argument_value(argument) == 3) {
            return 1;
        }
        std::cerr  << "Mesh is not a triangle mesh." << std::endl;
        return 0;
    }
    **face = ply_get_argument_value(argument);
    (*face)++;
    return 1;
}

void readPlyFile(Matrix& V, const char* filename,  bool readColor=true, bool readConfidence=true, bool readIntensity=true, bool readNormals=true, bool readFaces=true) {
    /* Start reading new PLY */
    p_ply ply = ply_open(filename, NULL, 0, NULL);

    if (!ply) {
        std::cerr  << "Could not open »" << filename << "«." << std::endl;
        return ;
    }
    if (!ply_read_header(ply)) {
        std::cerr  << "Could not read header." << std::endl;
        return;
    }
    std::cout  << "Loading »" << filename << "«." << std::endl;

    /* Check if there are vertices and get the amount of vertices. */
    char buf[256] = "";
    const char * name = buf;
    long int n;
    p_ply_element elem  = NULL;

    // Buffer count variables
    size_t numVertices              = 0;
    size_t numVertexColors          = 0;
    size_t numVertexConfidences     = 0;
    size_t numVertexIntensities     = 0;
    size_t numVertexNormals         = 0;

    size_t numPoints                = 0;
    size_t numPointColors           = 0;
    size_t numPointConfidence       = 0;
    size_t numPointIntensities      = 0;
    size_t numPointNormals          = 0;
    size_t numFaces                 = 0;

    while ((elem = ply_get_next_element(ply, elem))) {
        ply_get_element_info(elem, &name, &n);
        if (!strcmp(name, "vertex")) {
            numVertices = n;
            p_ply_property prop = NULL;
            while ((prop = ply_get_next_property(elem, prop))) {
                ply_get_property_info(prop, &name, NULL, NULL, NULL);
                if (!strcmp(name, "red") && readColor) {
                    /* We have color information */
                    numVertexColors = n;
                }
                else if (!strcmp(name, "confidence") && readConfidence ) {
                    /* We have confidence information */
                    numVertexConfidences = n;
                }
                else if (!strcmp(name, "intensity") && readIntensity) {
                    /* We have intensity information */
                    numVertexIntensities = n;
                }
                else if (!strcmp(name, "nx") && readNormals) {
                    /* We have normals */
                    numVertexNormals = n;
                }
            }
        }
        else if (!strcmp(name, "point")) {
            numPoints = n;
            p_ply_property prop = NULL;
            while ((prop = ply_get_next_property(elem, prop))) {
                ply_get_property_info(prop, &name, NULL, NULL, NULL);
                if (!strcmp(name, "red") && readColor) {
                    /* We have color information */
                    numPointColors = n;
                }
                else if (!strcmp(name, "confidence") && readConfidence) {
                    /* We have confidence information */
                    numPointConfidence = n;
                }
                else if (!strcmp( name, "intensity") && readIntensity) {
                    /* We have intensity information */
                    numPointIntensities = n;
                }
                else if (!strcmp(name, "nx") && readNormals) {
                    /* We have normals */
                    numPointNormals = n;
                }
            }
        }
        else if (!strcmp(name, "face") && readFaces) {
            numFaces = n;
        }
    }

    if (!(numVertices || numPoints)) {
        std::cout << "Neither vertices nor points in ply." << std::endl;
        return ;
    }

    // Buffers
    floatArr vertices;
    floatArr vertexConfidence;
    floatArr vertexIntensity;
    floatArr vertexNormals;
    floatArr points;
    floatArr pointConfidences;
    floatArr pointIntensities;
    floatArr pointNormals;

    ucharArr pointColors;
    ucharArr vertexColors;
    uintArr  faceIndices;


    /* Allocate memory. */
    if (numVertices) {
        vertices = floatArr(new float[numVertices * 3]);
    }
    if (numVertexColors) {
        vertexColors = ucharArr(new unsigned char[numVertices * 3]);
    }
    if (numVertexConfidences) {
        vertexConfidence = floatArr(new float[numVertices]);
    }
    if (numVertexIntensities) {
        vertexIntensity = floatArr(new float[numVertices]);
    }
    if (numVertexNormals) {
        vertexNormals = floatArr(new float[3 * numVertices]);
    }
    if (numFaces) {
        faceIndices = uintArr( new unsigned int[numFaces * 3]);
    }
    if (numPoints) {
        points = floatArr(new float[numPoints * 3]);
    }
    if (numPointColors) {
        pointColors = ucharArr(new unsigned char[numPoints * 3]);
    }
    if (numPointConfidence) {
        pointConfidences = floatArr(new float[numPoints]);
    }
    if (numPointIntensities) {
        pointIntensities = floatArr(new float[numPoints]);
    }
    if (numPointNormals) {
        pointNormals = floatArr(new float[3 * numPoints]);
    }


    float*        vertex            = vertices.get();
    uint8_t* 	  vertex_color      = vertexColors.get();
    float*        vertex_confidence = vertexConfidence.get();
    float*        vertex_intensity  = vertexIntensity.get();
    float*        vertex_normal     = vertexNormals.get();
    unsigned int* face              = faceIndices.get();
    float*        point             = points.get();
    uint8_t*      point_color       = pointColors.get();
    float*        point_confidence  = pointConfidences.get();
    float*        point_intensity   = pointIntensities.get();
    float*        point_normal      = pointNormals.get();


    /* Set callbacks. */
    if (vertex) {
        ply_set_read_cb(ply, "vertex", "x", readVertexCb, &vertex, 0);
        ply_set_read_cb(ply, "vertex", "y", readVertexCb, &vertex, 0);
        ply_set_read_cb(ply, "vertex", "z", readVertexCb, &vertex, 1);
    }
    if (vertex_color) {
        ply_set_read_cb(ply, "vertex", "red",   readColorCb,  &vertex_color,  0);
        ply_set_read_cb(ply, "vertex", "green", readColorCb,  &vertex_color,  0);
        ply_set_read_cb(ply, "vertex", "blue",  readColorCb,  &vertex_color,  1);
    }
    if (vertex_confidence) {
        ply_set_read_cb(ply, "vertex", "confidence", readVertexCb, &vertex_confidence, 1);
    }
    if (vertex_intensity) {
        ply_set_read_cb(ply, "vertex", "intensity", readVertexCb, &vertex_intensity, 1);
    }
    if (vertex_normal) {
        ply_set_read_cb(ply, "vertex", "nx", readVertexCb, &vertex_normal, 0);
        ply_set_read_cb(ply, "vertex", "ny", readVertexCb, &vertex_normal, 0);
        ply_set_read_cb(ply, "vertex", "nz", readVertexCb, &vertex_normal, 1);
    }

    if (face) {
        ply_set_read_cb(ply, "face", "vertex_indices", readFaceCb, &face, 0);
        ply_set_read_cb(ply, "face", "vertex_index", readFaceCb, &face, 0);
    }

    if (point) {
        ply_set_read_cb(ply, "point", "x", readVertexCb, &point, 0);
        ply_set_read_cb(ply, "point", "y", readVertexCb, &point, 0);
        ply_set_read_cb(ply, "point", "z", readVertexCb, &point, 1);
    }
    if (point_color) {
        ply_set_read_cb(ply, "point", "red",   readColorCb,  &point_color,  0);
        ply_set_read_cb(ply, "point", "green", readColorCb,  &point_color,  0);
        ply_set_read_cb(ply, "point", "blue",  readColorCb,  &point_color,  1);
    }
    if (point_confidence) {
        ply_set_read_cb(ply, "point", "confidence", readVertexCb, &point_confidence, 1);
    }
    if (point_intensity) {
        ply_set_read_cb(ply, "point", "intensity", readVertexCb, &point_intensity, 1);
    }
    if (point_normal) {
        ply_set_read_cb(ply, "point", "nx", readVertexCb, &point_normal, 0);
        ply_set_read_cb(ply, "point", "ny", readVertexCb, &point_normal, 0);
        ply_set_read_cb(ply, "point", "nz", readVertexCb, &point_normal, 1);
    }

    /* Read ply file. */
    if (!ply_read(ply)) {
        std::cerr << "Could not read »" << filename << "«." << std::endl;
    }

    ply_close(ply);

    V.width = numVertices;
    V.height = 3;
    V.elements = (float*)malloc(V.width * V.height * sizeof(float));

    if(vertices) {
        for(int i=0; i<numVertices; i++) {
            V.elements[V.width*0+i] = vertices[i*3+0];
            V.elements[V.width*1+i] = vertices[i*3+1];
            V.elements[V.width*2+i] = vertices[i*3+2];
        }
    }

    printf("finish reading\n");
}

void writePlyFile(const Matrix& V, const Matrix& Result_Normals, const char* filename) {
    /* Handle options. */
    e_ply_storage_mode mode(PLY_LITTLE_ENDIAN);

    p_ply oply = ply_create( filename, mode, NULL, 0, NULL);
    if (!oply) {
        std::cerr  << "Could not create »" << filename << "«" << std::endl;
        return;
    }

    size_t m_numVertices = V.width;
    size_t m_numVertexNormals = Result_Normals.width;

    float* m_vertices = new float[m_numVertices * 3];
    float* m_vertexNormals = new float[m_numVertexNormals * 3];

    for(size_t i=0; i<V.width; i++) {
        m_vertices[i * 3] = V.elements[i];
        m_vertices[i * 3 + 1] = V.elements[V.width + i];
        m_vertices[i * 3 + 2] = V.elements[2 * V.width + i];

        m_vertexNormals[i * 3] = Result_Normals.elements[i];
        m_vertexNormals[i * 3 + 1] = Result_Normals.elements[Result_Normals.width + i];
        m_vertexNormals[i * 3 + 2] = Result_Normals.elements[2 * Result_Normals.width + i];
    }

    bool vertex_normal = false;

    /* Add vertex element. */
    if (m_vertices) {
        ply_add_element(oply, "vertex", m_numVertices);

        /* Add vertex properties: x, y, z, (r, g, b) */
        ply_add_scalar_property(oply, "x", PLY_FLOAT);
        ply_add_scalar_property(oply, "y", PLY_FLOAT);
        ply_add_scalar_property(oply, "z", PLY_FLOAT);

        /* Add normals if there are any. */
        if (m_vertexNormals) {

            if (m_numVertexNormals != m_numVertices) {
                std::cout << "Amount of vertices and normals" << " does not match. Normals won't be written." << std::endl;
            }
            else {
                ply_add_scalar_property(oply, "nx", PLY_FLOAT);
                ply_add_scalar_property(oply, "ny", PLY_FLOAT);
                ply_add_scalar_property(oply, "nz", PLY_FLOAT);
                vertex_normal = true;
            }
        }
    }

    /* Write header to file. */
    if (!ply_write_header(oply)) {
        std::cerr  << "Could not write header." << std::endl;
        return;
    }

    for (size_t i = 0; i < m_numVertices; i++ ) {
        ply_write(oply, (double) m_vertices[i * 3    ]); /* x */
        ply_write(oply, (double) m_vertices[i * 3 + 1]); /* y */
        ply_write(oply, (double) m_vertices[i * 3 + 2]); /* z */

        if (vertex_normal) {
            ply_write(oply, (double) m_vertexNormals[i * 3    ]); /* nx */
            ply_write(oply, (double) m_vertexNormals[i * 3 + 1]); /* ny */
            ply_write(oply, (double) m_vertexNormals[i * 3 + 2]); /* nz */
        }
    }

    if (!ply_close(oply)) {
        std::cerr  << "Could not close file." << std::endl;
    }

    printf("finish writing\n");
}

void reducePointCloud(Matrix& All_Points, Matrix& V, int target_size){
    V.width = target_size;
    V.height = 3;
    mallocMat(V);

    for(int i=0; i<target_size; i++){
        int index = (int)((float)rand() * ((float)All_Points.width-1.0)/RAND_MAX   ) ;

        V.elements[V.width*0+i] = All_Points.elements[All_Points.width*0+index];
        V.elements[V.width*1+i] = All_Points.elements[All_Points.width*1+index];
        V.elements[V.width*2+i] = All_Points.elements[All_Points.width*2+index];

    }
}

void kNearestNeighborSearch(int k, int numPoints, const char* file, const char* dest) {
    Matrix pointMat;
    pointMat.width = numPoints;
    pointMat.height = 3;
    mallocMat(pointMat);

    // Init point matrix from file or at random
    if (file) {
        Matrix pointMatAll;
        readPlyFile(pointMatAll, file);
        reducePointCloud(pointMatAll, pointMat, numPoints);
        free(pointMatAll.elements);
    } else {
        initMat(pointMat);
    }

    Matrix idxVec;
    idxVec.width = k;
    idxVec.height = 1;
    mallocMat(idxVec);

    Matrix neighbMat;
    neighbMat.width = k;
    neighbMat.height = 3;
    mallocMat(neighbMat);

    Matrix normalMat;
    normalMat.width = pointMat.width;
    normalMat.height = pointMat.height;
    mallocMat(normalMat);

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

    float epsilon = 5;
    int progress = 0;

    // Iterate over all points
    for (int i = 0; i < pointMat.width; i++) {
        // Calc distances to all other points
        getDistances(d_pointMat, d_distVec, i);

        // Evaluate threshold
        evalEpsilon(d_distVec, numNeighb, d_numNeighb, k, epsilon, numPoints/20000);

        // Get k nearest neighbors
        getNearestNeighbors(d_distVec, d_idxVec, epsilon);
        cudaMemcpy(idxVec.elements, d_idxVec.elements, size, cudaMemcpyDeviceToHost);

        // Get direction vectors to nearest neighbors
        getDirections(idxVec, pointMat, neighbMat, i);

        // Calc normal
        calcNormals(neighbMat, normalMat, i, 10);

        // Print out progress
        progress = (100 * (i + 1)) / pointMat.width;
        cout << "Progress: " << progress << "%\r";
        cout.flush();
    }

    if(file)
        writePlyFile(pointMat, normalMat, dest);

    // Free device memory
    cudaFree(d_numNeighb);
    cudaFree(d_idxVec.elements);
    cudaFree(d_distVec.elements);
    cudaFree(d_pointMat.elements);

    free(numNeighb);
    free(normalMat.elements);
    free(neighbMat.elements);
    free(idxVec.elements);
    free(pointMat.elements);
}
