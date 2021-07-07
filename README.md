<h1>DEPRECATED</h1>

Moved to Las-Vegas-Reconstruction-Toolkit (Surface Reconstruction from PointClouds). Link: https://github.com/uos/lvr2. 

Improvements:
- Bugfixes
- Improved Cuda implementation
- OpenCl port

LVR2 builds the executables `lvr2_cuda_normals` and `lvr2_cl_normals` which use normal the estimation on GPU in minimal form.


<h1>OLD Softwarename: cudaNormals</h1>
<br>
<h1> k - Nearest Neighbor Search on Pointclouds for normal estimation with CUDA. </h1>

<h3>Requirements:</h3>
<ul>
<li> CUDA (Minimum 8.0) (https://developer.nvidia.com/cuda-toolkit) </li>
<li> CMake (Minimum 2.8.8) </li>
<li> Boost Library for the example main.cpp </li>
</ul>

<h3>Abstract</h3>
<p>
This Toolkit includes a class for normal calculation on big pointclouds. It was researched during the "Parallel computing"-course of the University Osnabrück (Lecturer: Juan Carlos Saborio Morales). The software generates a left-balanced array-based kd-tree on a pointcloud. The k-nearest neighbor search and the normal calculation is computed highly parallel on the GPU with the CUDA-Toolkit. 
</p>

<h3>Workflow</h3>
![alt tag](https://github.com/aock/kNNcuda/blob/master/res/workflow_normals_cuda.png)


<br>
<h2>Normal Calculation with CUDA:</h2>

<h3>Include calcNormalsCuda.h in your program. </h3>
<p>Add your code to the example CMakeLists.txt:</p>
<pre><code>    ##### ADD YOUR CODE HERE #####
    add_executable(your_program path/to/your/code.cpp)
    
    target_link_libraries(your_program
        normalsCuda
    )
</code></pre>

<p>Header:</p>
<p>include/calcNormalsCuda.h</p>


<h3>Parameters in constructor are:</h3>
- Pointcloud (as struct "PointArray" defined in calcNormalsCuda.h)
<p>constructs the kd-tree</p>


<h3>Set other global parameters with:</h3>
- setK(int): set the k of the kNN search. Default: 50
- setFlippoint(float x, float y, float z): set the normal orientation point for flipping. Default: (100000.0, 100000.0, 100000.0).
- setMethod(const char* method): set method for normal calculation. "RANSAC"/"PCA". Default: "PCA". 


<h3>Start normal calculation on GPU</h3>
- start()


<h3>Get resulting normals</h3>
- getNormals(PointArray& normals): mallocs and fills the resulting normal array.


<p>Struct PointArray:</p>
<pre><code>struct PointArray { <br>
        int width; <br>
        int dim; <br>
        float* elements; <br>
    };
</code></pre>

<h3>Example Code (src/main.cpp)</h3>
<p>Reading a File with the Stanford Triangle Format (.ply) to a PointArray</p>
<p>Executing the CUDA normal calculation</p>
<p>Writing PointArray of points and PointArray of normals to destination file</p>





