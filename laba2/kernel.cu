#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include"SaveData.h"


__global__ void waveKernel(float* p_np1, float* p_n, float* p_nm1, int Nx, int Nz, float v, float dt, float dx, float dz, int is, int js, float S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 5;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 5;
    if (i <= Nx-4 && j <= Nz-4) {
        int idx = i * (Nz + 2) + j;
        float d2p_dx2 = (-1.0f / 560 * p_n[idx - 4 * (Nz + 2)] + 8.0f / 315 * p_n[idx - 3 * (Nz + 2)] - 1.0f / 5 * p_n[idx - 2 * (Nz + 2)] +
            8.0f / 5 * p_n[idx - (Nz + 2)] - 205.0f / 72 * p_n[idx] + 8.0f / 5 * p_n[idx + (Nz + 2)] -
            1.0f / 5 * p_n[idx + 2 * (Nz + 2)] + 8.0f / 315 * p_n[idx + 3 * (Nz + 2)] - 1.0f / 560 * p_n[idx + 4 * (Nz + 2)]) / (dx * dx);
        float d2p_dz2 = (-1.0f / 560 * p_n[idx - 4] + 8.0f / 315 * p_n[idx - 3] - 1.0f / 5 * p_n[idx - 2] +
            8.0f / 5 * p_n[idx - 1] - 205.0f / 72 * p_n[idx] + 8.0f / 5 * p_n[idx + 1] -
            1.0f / 5 * p_n[idx + 2] + 8.0f / 315 * p_n[idx + 3] - 1.0f / 560 * p_n[idx + 4]) / (dz * dz);
        p_np1[idx] = 2 * p_n[idx] - p_nm1[idx] + v * v * dt * dt * (d2p_dx2 + d2p_dz2);
        if (i == is && j == js) p_np1[idx] += v * v * dt * dt * S;
    }
}
/*
int main() {
    const int Nx = 299, Nz = 299, Nt = 1000;
    const float v = 3000.0, dt = 0.001, dx = 10.0, dz = 10.0;
    const float f0 = 15.0, t0 = 2.0 / f0;
    const int is = Nx / 2, js = Nz / 2;
    const int size = (Nx + 2) * (Nz + 2) * sizeof(float);

    float* d_p_nm1, * d_p_n, * d_p_np1;
    float* h_p = new float[(Nx + 2) * (Nz + 2)];
    cudaMalloc(&d_p_nm1, size); cudaMalloc(&d_p_n, size); cudaMalloc(&d_p_np1, size);
    cudaMemset(d_p_nm1, 0, size); cudaMemset(d_p_n, 0, size); cudaMemset(d_p_np1, 0, size);

    dim3 block(16, 16);
    dim3 grid((Nx + block.x - 1) / block.x, (Nz + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int n = 0; n < Nt; ++n) {
        float t = n * dt;
        float S = (1 - 2 * powf(M_PI * f0 * (t - t0), 2)) * expf(-powf(M_PI * f0 * (t - t0), 2));
        waveKernel << <grid, block >> > (d_p_np1, d_p_n, d_p_nm1, Nx, Nz, v, dt, dx, dz, is, js, S);
        cudaMemset(d_p_np1, 0, (Nz + 2) * sizeof(float)); // Граничные условия
        cudaMemset(d_p_np1 + (Nx + 1) * (Nz + 2), 0, (Nz + 2) * sizeof(float));
        
        //for (int j = 0; j <= Nz + 1; ++j) {
        //    float* left = d_p_np1 + j;
        //    float* right = d_p_np1 + (Nx + 1) * (Nz + 2) + j;
        //    cudaMemset(left, 0, sizeof(float)); // Левая граница
        //    cudaMemset(right, 0, sizeof(float)); // Правая граница
        //}
        
        float* temp = d_p_nm1; d_p_nm1 = d_p_n; d_p_n = d_p_np1; d_p_np1 = temp;
        
        if (n % 100 == 0) {
            cudaMemcpy(h_p, d_p_n, size, cudaMemcpyDeviceToHost);
            // Создаем вектор векторов и копируем данные
            std::vector<std::vector<double>> p(Nx + 2, std::vector<double>(Nz + 2));
            for (int i = 0; i < Nx + 2; ++i) {
                for (int j = 0; j < Nz + 2; ++j) {
                    p[i][j] = h_p[i * (Nz + 2) + j];
                }
            }
            std::string filename = "resultsGPU_" + std::to_string(n / 100) + ".csv";
            saveResults(p, filename);

        }
        
    }

    
   // cudaMemcpy(h_p, d_p_n, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "CUDA time: " << time / 1000 << " s\n";
    cudaFree(d_p_nm1); cudaFree(d_p_n); cudaFree(d_p_np1);
    delete[] h_p;
    return 0;
}*/