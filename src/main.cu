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
    std::cout<<"\n";


    std::cout<<"Adding the above two matrices"<<std::endl;
    std::cout<<"\n";
    z2=z+z1;
    printmatrix(z2);
    std::cout<<"\n";

    std::cout<<"Multiplying Z and Z1 matrices"<<std::endl;
    std::cout<<"\n";
    z3=GEMM(z,z1);
    printmatrix(z3);
    
    
    
    
    return 0;
}