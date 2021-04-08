#include<curand.h>
#include <cublas_v2.h>
#include"matrices.cu"

template<typename T>
T* gpumatrix(Matrix<T>& other)
{
    //Convert a CPU matrix to GPU Matrix
    T * gdata;
    cudaMalloc(&gdata,sizeof(T)*other.m*other.n);
    cudaMemcpy(&gdata,&(other.M),sizeof(T)*other.m*other.n,cudaMemcpyHostToDevice);
    return gdata;
}
/*template<typename T>
 GEMM(Matrix<T>& other,Matrix<T>& other2)
{
  Matrix<T>& c(other.m,other.n);
  
  


}*/
int main()
{
    Matrix<int> z(2,6);
    Matrix<int> z1(2,6);
    int * ls=gpumatrix(z);
    int * ls1=gpumatrix(z1);
    return 0;
}