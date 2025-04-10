#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include"SaveData.h"
#define M_PI 3.14159265
__global__ void waveSharedKernel(float* p_np1, float* p_n, float* p_nm1, int Nx, int Nz, float v, float dt, float dx, float dz, int is, int js, float S) {
    extern __shared__ float s_p[];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x, by = blockIdx.y * blockDim.y;
    int i = bx + tx + 1, j = by + ty + 1;
    int s_width = blockDim.x + 8;

    // Загрузка в Shared Memory
    if (i <= Nx + 1 && j <= Nz + 1) {
        int g_idx = i * (Nz + 2) + j;
        int s_idx = (tx + 4) * s_width + (ty + 4);
        s_p[s_idx] = p_n[g_idx];
        if (tx < 4) s_p[(tx)*s_width + (ty + 4)] = p_n[(bx + tx - 3) * (Nz + 2) + j];
        if (ty < 4) s_p[(tx + 4) * s_width + ty] = p_n[i * (Nz + 2) + (by + ty - 3)];
        if (tx >= blockDim.x - 4) s_p[(tx + 8) * s_width + (ty + 4)] = p_n[(bx + tx + 5) * (Nz + 2) + j];
        if (ty >= blockDim.y - 4) s_p[(tx + 4) * s_width + (ty + 8)] = p_n[i * (Nz + 2) + (by + ty + 5)];
    }
    __syncthreads();

    if (i >= 5 && j >= 5 && i <= Nx-4 && j <= Nz-4) {
        int s_idx = (tx + 4) * s_width + (ty + 4);
        float d2p_dx2 = (-1.0f / 560 * s_p[s_idx - 4 * s_width] + 8.0f / 315 * s_p[s_idx - 3 * s_width] - 1.0f / 5 * s_p[s_idx - 2 * s_width] +
            8.0f / 5 * s_p[s_idx - s_width] - 205.0f / 72 * s_p[s_idx] + 8.0f / 5 * s_p[s_idx + s_width] -
            1.0f / 5 * s_p[s_idx + 2 * s_width] + 8.0f / 315 * s_p[s_idx + 3 * s_width] - 1.0f / 560 * s_p[s_idx + 4 * s_width]) / (dx * dx);
        float d2p_dz2 = (-1.0f / 560 * s_p[s_idx - 4] + 8.0f / 315 * s_p[s_idx - 3] - 1.0f / 5 * s_p[s_idx - 2] +
            8.0f / 5 * s_p[s_idx - 1] - 205.0f / 72 * s_p[s_idx] + 8.0f / 5 * s_p[s_idx + 1] -
            1.0f / 5 * s_p[s_idx + 2] + 8.0f / 315 * s_p[s_idx + 3] - 1.0f / 560 * s_p[s_idx + 4]) / (dz * dz);
        int g_idx = i * (Nz + 2) + j;
        p_np1[g_idx] = 2 * p_n[g_idx] - p_nm1[g_idx] + v * v * dt * dt * (d2p_dx2 + d2p_dz2);
        if (i == is && j == js) p_np1[g_idx] += v * v * dt * dt * S;
    }
}

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
    size_t shared_size = (block.x + 8) * (block.y + 8) * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int n = 0; n < Nt; ++n) {
        float t = n * dt;
        float S = (1 - 2 * powf(M_PI * f0 * (t - t0), 2)) * expf(-powf(M_PI * f0 * (t - t0), 2));
        waveSharedKernel << <grid, block, shared_size >> > (d_p_np1, d_p_n, d_p_nm1, Nx, Nz, v, dt, dx, dz, is, js, S);
        cudaMemset(d_p_np1, 0, (Nz + 2) * sizeof(float));
        cudaMemset(d_p_np1 + (Nx + 1) * (Nz + 2), 0, (Nz + 2) * sizeof(float));
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

    
    cudaMemcpy(h_p, d_p_n, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "CUDA Shared time: " << time / 1000 << " s\n";

    cudaFree(d_p_nm1); cudaFree(d_p_n); cudaFree(d_p_np1);
    delete[] h_p;
    return 0;
}