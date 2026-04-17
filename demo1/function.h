#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <vector>
using namespace std;
// Initialize matrix with random values
void initializeMatrix(vector<vector<double>>& matrix, int rows, int cols);
// Initialize vector with random values
void initializeVector(vector<double>& vec, int size);
// Sequential matrix-vector multiplication
void matrixVectorMultSeq(const vector<vector<double>>& matrix, 
                         const vector<double>& vec, 
                         vector<double>& result);
// Parallel matrix-vector multiplication using OpenMP
void matrixVectorMultPar(const vector<vector<double>>& matrix, 
                         const vector<double>& vec, 
                         vector<double>& result);
// Print vector (for verification)
void printVector(const vector<double>& vec, int limit = 10);
#endif