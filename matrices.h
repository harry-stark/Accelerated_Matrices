#include <iostream>
#include <stdlib.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16
using namespace std;
#ifndef MATRICES_H
#define MATRICES_H

template <typename T>
class Matrix
{
public:
    int m;
    int n;
    T *M;
    Matrix(int rows, int column);
    Matrix() : m(1), n(1) {}
    ~Matrix();
    //Matrix(Matrix<T> &const other);
    Matrix operator+(Matrix<T> const other);
    Matrix &operator=(Matrix<T> const &other);
    Matrix operator-(Matrix<T> const other);
};

template <typename T>
Matrix<T>::Matrix(int rows, int columns) : m(rows), n(columns)
{
    M = new T[rows * columns];
}

template <typename T>
Matrix<T>::~Matrix(){}

template <typename T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> const &other)
{

    if ((this->m > 0) && (this->n > 0))
    {
        delete[] M;
    }
    this->m = other.m;
    this->n = other.n;

    this->M = new T[other.m * other.n];

    for (int i = 0; i < other.m; i++)
    {
        for (int j = 0; j < other.n; j++)
        {
            this->M[i * other.n + j] = other.M[i * other.n + j];
        }
    }

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(Matrix<T> const other)
{

    if ((this->m == other.m) && this->n == other.n)
    {
        for (int i = 0; i < other.m; i++)
        {
            for (int j = 0; j < other.n; j++)
            {
                //cout<<"We are going to add "<<this->M[i*other.n+j]<<" and "<<other.M[i*other.n+j]<<"\n";
                this->M[i * other.n + j] = this->M[i * other.n + j] + other.M[i * other.n + j];
                //cout<<"Added Value: "<<this->M[i*other.n+j]<<"\n";
            }
        }
    }
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(Matrix<T> const other)
{

    if ((this->m == other.m) && this->n == other.n)
    {
        for (int i = 0; i < other.m; i++)
        {
            for (int j = 0; j < other.n; j++)
            {
                //cout<<"We are going to add "<<this->M[i*other.n+j]<<" and "<<other.M[i*other.n+j]<<"\n";
                this->M[i * other.n + j] = this->M[i * other.n + j] - other.M[i * other.n + j];
                //cout<<"Added Value: "<<this->M[i*other.n+j]<<"\n";
            }
        }
    }
    return *this;
}

template <typename T>
void fill(Matrix<T> &other)
{
    for (int i = 0; i < other.m; i++)
    {
        for (int j = 0; j < other.n; j++)
        {
            other.M[i * other.n + j] = rand() % 100;
        }
    }
}

template <typename T>
void printmatrix(Matrix<T> &other)
{
    for (int i = 0; i < other.m; i++)
    {
        for (int j = 0; j < other.n; j++)
        {
            cout << other.M[i * other.n + j] << "\t";
        }
        cout << "\n";
    }
    cout << "---------------------------------"
         << "\n";
}

__global__ void matrixMul(const int *a, const int *b, int *c, int m, int n, int k)
{
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

template <typename T>
Matrix<T> GEMM(Matrix<T> &other, Matrix<T> &other1)
{

    T *gdata_a;
    T *gdata_b;
    T *gdata_c;

    const size_t bytes_a = other.m * other.n * sizeof(T);
    const size_t bytes_b = other1.m * other1.n * sizeof(T);
    const size_t bytes_c = other.m * other1.n * sizeof(T);

    //allocate memory

    cudaMalloc(&gdata_a, bytes_a);
    cudaMalloc(&gdata_b, bytes_b);
    cudaMalloc(&gdata_c, bytes_c);
    //Copy from host to device
    cudaMemcpy(gdata_a, (other.M), bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(gdata_b, (other1.M), bytes_b, cudaMemcpyHostToDevice);
    Matrix<T> other2(other.m, other1.n);

    if constexpr (std::is_same_v<T, int>)
    {

        unsigned int grid_rows = (other2.m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (other2.n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        matrixMul<<<dimGrid, dimBlock>>>(gdata_a, gdata_b, gdata_c, other.m, other.n, other1.n);
        cudaMemcpy(other2.M, gdata_c, bytes_c, cudaMemcpyDeviceToHost);
        cudaFree(gdata_a);
        cudaFree(gdata_b);
        cudaFree(gdata_c);
    }

    else
    {

        // cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Scalaing factors
        float alpha = 1.0f;
        float beta = 0.0f;

        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, other.m, other.n, other1.n, &alpha, gdata_a, other.m, gdata_b, other.n, &beta, gdata_c, other.m);

        // Copy back the three matrices
        cudaMemcpy(other2.M, gdata_c, bytes_c, cudaMemcpyDeviceToHost);

        // free gpu memory
        cudaFree(gdata_a);
        cudaFree(gdata_b);
        cudaFree(gdata_c);
    }
    return other2;
}
#endif