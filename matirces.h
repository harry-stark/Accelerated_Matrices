#include<iostream>

template<typename T>
class Matrix{
    public:
    int m;
    int n;
    T *M;
    Matrix(int rows, int column);
    Matrix():m(1),n(1){}
    ~Matrix();
    Matrix(Matrix<T>& const other);
    Matrix operator=(Matrix<T>& const other);
    Matrix operator+(Matrix<T>& const other);
    Matrix operator-(Matrix<T>& const other);

};

