#include"matrices.cu"


int main()
{
    
    Matrix<float> z(4,4);
    Matrix<float> z1(4,4);
    Matrix<float> z2(4,4);
    //Matrix<float> z3(4,4);
    fill(z);
    fill(z1);
    printmatrix(z);
    printmatrix(z1);
    z2=z+z1;
    printmatrix(z2);
    //std::cout<<"Delta"<<std::endl;
    //z3=GEMM(z,z1);
    //printmatrix(z3);
    
    
    return 0;
}