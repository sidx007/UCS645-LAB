#include "functions.h"
#include <iostream>
#include <random>
#include <omp.h>

using namespace std;

void initializeMatrix(vector<vector<double>>& matrix, int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = dis(gen);
        }
    }
}

void initializeVector(vector<double>& vec, int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0);
    
    for (int i = 0; i < size; i++) {
        vec[i] = dis(gen);
    }
}

void matrixVectorMultSeq(const vector<vector<double>>& matrix, 
                         const vector<double>& vec, 
                         vector<double>& result) {
    int rows = matrix.size();
    int cols = vec.size();
    
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
}

void matrixVectorMultPar(const vector<vector<double>>& matrix, 
                         const vector<double>& vec, 
                         vector<double>& result) {
    int rows = matrix.size();
    int cols = vec.size();
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
}

void printVector(const vector<double>& vec, int limit) {
    cout << "[";
    int size = vec.size();
    int printSize = (size < limit) ? size : limit;
    
    for (int i = 0; i < printSize; i++) {
        cout << vec[i];
        if (i < printSize - 1) cout << ", ";
    }
    
    if (size > limit) {
        cout << " ... (" << size << " elements total)";
    }
    cout << "]" << endl;
}