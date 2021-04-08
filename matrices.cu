#include<iostream>
#include <stdlib.h>
#include <cublas_v2.h>
using namespace std;

template<typename T>
class Matrix{
public:
    T* M;
    int m=1;
    int n=1;

Matrix(int rows , int columns ):m(rows),n(columns)
{
    M = new T[rows*columns];
}
Matrix operator+(Matrix<T>& other);
Matrix& operator=(Matrix<T> const &other);
/*
{

    if((this->m>0)&&(this->n>0))
    {
        delete this->M;
    }
    this->m=other.m;
    this->n=other.n;

    this->M = new T[other.m*other.n];

    for(int i=0;i<other.m;i++){
        for(int j=0;j<other.n;j++){
            this->M[i*other.n+j]=other.M[i*other.n+j];
        }
    }

    return *this;
}*/


~Matrix(){
    if(m>0 || n>0)
    {
        delete  M;
    }
}
};


template<typename T>
void fill(Matrix<T> &other)
{
    for(int i=0;i<other.m;i++)
    {
        for(int j=0;j<other.n;j++)
        {
            other.M[i*other.n+j]=rand() % 100;
        }
    }
}

template<typename T>
void printmatrix(Matrix<T> &other)
{
    for(int i=0;i<other.m;i++)
    {
      for(int j=0;j<other.n;j++)
      {
          cout<<other.M[i*other.n+j]<<"\t";
      }
      cout<<"\n";
    }
    cout<<"---------------------------------"<<"\n";
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T> const & other){

    if((this->m>0)&&(this->n>0))
    {
        delete this->M;
    }
    this->m=other.m;
    this->n=other.n;

    this->M = new T[other.m*other.n];

    for(int i=0;i<other.m;i++){
        for(int j=0;j<other.n;j++){
            this->M[i*other.n+j]=other.M[i*other.n+j];
        }
    }

    return *this;
}

__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    // Iterate over row, and down column
    c[row * N + col] = 0;
    for (int k = 0; k < N; k++) {
      // Accumulate results for a single element
      c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
  }
  

template<typename T>
Matrix<T> Matrix<T>::operator+(Matrix<T>& const other){

    if((this->m==other.m)&& this->n==other.n){
        for(int i=0;i<other.m;i++)
        {
            for(int j=0;j<other.n;j++){
                //cout<<"We are going to add "<<this->M[i*other.n+j]<<" and "<<other.M[i*other.n+j]<<"\n";
                this->M[i*other.n+j]=this->M[i*other.n+j]+other.M[i*other.n+j];
                //cout<<"Added Value: "<<this->M[i*other.n+j]<<"\n";
            }
        }
    }
    return *this;

}
template<typename T>
Matrix<T> GEMM(Matrix<T>& other,Matrix<T>& other1){

    T* gdata_a;
    T* gdata_b;
    T* gdata_c;
    
    const size_t bytes_a = other.m* other.n * sizeof(T);
    const size_t bytes_b = other1.m * other1.n * sizeof(T);
    const size_t bytes_c = other.m* other1.n * sizeof(T);

    

    //allocate memory

    cudaMalloc(&gdata_a,bytes_a);
    cudaMalloc(&gdata_b,bytes_b);
    cudaMalloc(&gdata_c,bytes_c);
    //Copy from host to device
    cudaMemcpy(&gdata_a,&(other.M),bytes_a,cudaMemcpyHostToDevice);
    cudaMemcpy(&gdata_b,&(other1.M),bytes_b,cudaMemcpyHostToDevice);

    //cudaMemcpy(&gdata_c,&(other2.M),bytes_c,cudaMemcpyHostToDevice);
 /*if(T==int)
    {


        Matrix<T> other2(other.m,other1.n);
        cudaMemcpy(other2.M, gdata_c, bytes_c, cudaMemcpyDeviceToHost);
        int THREADS = 32;
        int BLOCKS = / THREADS;


    }*/

 //else if
 {
    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scalaing factors
    float alpha = 1.0f;
    float beta = 0.0f;
  
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, other.m, other.n, other1.n, &alpha, gdata_a, other.m, gdata_b, other.n,&beta, gdata_c, other.m);
     
    Matrix<T> other2(other.m,other1.n);
    // Copy back the three matrices
    cudaMemcpy(other2.M, gdata_c, bytes_c, cudaMemcpyDeviceToHost);
    // 
    cudaFree(gdata_a);
    cudaFree(gdata_b);
    cudaFree(gdata_c);


    return other2;
 }



}
