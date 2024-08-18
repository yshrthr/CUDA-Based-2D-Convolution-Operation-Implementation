/**
*   CS6023: GPU Programming
*   Assignment 2
*
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree()
*   to free up memory as soon as you're done with an allocation.
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;
__global__
void dkernel(long int* d_mat, long int *d_filter, long int *d_ans, int* matrix_row, int* matrix_col ,int* filter_size){
  int row = blockIdx.x;
  int col = threadIdx.x;
  int k = *filter_size;
  int m = *matrix_row;
  int n = *matrix_col;

  //printf("row: %d, col: %d, %ld\n ", row, col, d_mat[row * n + col]);
  long int sum = 0;
  for(int k1 = 0, i = row - (k/2); k1 < k && i <= row + (k/2); k1++, i++){
    for(int k2 = 0, j = col - (k/2); k2 < k && j <= col + (k/2); k2++, j++){
      if(i >= 0 && i < m){
        if(j >= 0 && j < n){
          //printf("%ld, %ld\n" , d_filter[k1*k + k2], d_mat[i * n + j]);
          sum += d_filter[k1*k + k2] * d_mat[i * n + j];
        }
      }
    }
  }

  d_ans[row * n + col]  = sum;

}

int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];

    long int* d_mat;
    long int *d_filter;
    long int* d_ans;
    int* filter_size;
    int* matrix_row;
    int* matrix_col;

    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
    **/

    /****************************************************Start Here***********************************************************/
    cudaMalloc(&d_mat, (m * n) * sizeof(long int));
    cudaMalloc(&d_filter, (k * k) * sizeof(long int));
    cudaMalloc(&d_ans, (m * n) * sizeof(long int));
    cudaMalloc(&filter_size, sizeof(int));
    cudaMalloc(&matrix_row, sizeof(int));
    cudaMalloc(&matrix_col, sizeof(int));


    cudaMemcpy(d_mat, h_mat, m * n * sizeof(long int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, k * k * sizeof(long int),cudaMemcpyHostToDevice);
    cudaMemcpy(filter_size, &k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_row, &m, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_col, &n, sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch
    //printf("launching");
    dkernel<<<m,n>>>(d_mat, d_filter, d_ans, matrix_row, matrix_col, filter_size);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch

    cudaMemcpy(h_ans, d_ans, m * n * sizeof(long int), cudaMemcpyDeviceToHost);

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
    */



    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}
