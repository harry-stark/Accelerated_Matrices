#include "../include/matrix/matrices.h"


int main()
{
    
    Matrix<int> z(4,4);
    Matrix<int> z1(4,4);
    Matrix<int> z2(4,4);
    Matrix<int> z3(4,4);
    fill(z);
    fill(z1);
    printmatrix(z);
    printmatrix(z1);
    z2=z+z1;
    printmatrix(z2);
    printmatrix(z);
    printmatrix(z1);

    z3=GEMM(z,z1);
    printmatrix(z3);
    printmatrix(z);
    printmatrix(z1);
    
    
    
    return 0;
}