#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#define M_PI 3.14159265
#include <fstream>
#include <string>
#include"SaveData.h"

void solveWaveEquationSequential() {
    // ��������� ������
    const double v = 3000.0;       // �������� �����, �/�
    const double Lx = 3000.0;      // ������ ������� �� x, �
    const double Lz = 3000.0;      // ������ ������� �� z, �
    const double f0 = 15.0;        // ������������ ������� ��������, ��
    const double t_max = 1.0;      // ������������ ����� �������������, �
    const double dx = 10.0;        // ��� ����� �� x, �
    const double dz = 10.0;        // ��� ����� �� z, �
    const double dt = 0.001;       // ��� �� �������, �

    // ���������� ���������� ����� � �����
    const int Nx = static_cast<int>(Lx / dx) - 1;  // ���������� ���������� ����� �� x
    const int Nz = static_cast<int>(Lz / dz) - 1;  // ���������� ���������� ����� �� z
    const int Nt = static_cast<int>(t_max / dt);   // ���������� ��������� �����
    const int is = Nx / 2;         // ������ ��������� �� x (� ������)
    const int js = Nz / 2;         // ������ ��������� �� z (� ������)
    const double t0 = 2.0 / f0;    // ��������� ����� ��� �������� ������

    // ������������� �������� ��� ��������� ���� (� ������ ������)
    std::vector<std::vector<double>> p_nm1(Nx + 2, std::vector<double>(Nz + 2, 0.0));  // p^{n-1}
    std::vector<std::vector<double>> p_n(Nx + 2, std::vector<double>(Nz + 2, 0.0));    // p^{n}
    std::vector<std::vector<double>> p_np1(Nx + 2, std::vector<double>(Nz + 2, 0.0));  // p^{n+1}

    // ������������ ��� ������������� ������ ����������� �������� �������
    const double c0 = -205.0 / 72.0;
    const double c1 = 8.0 / 5.0;
    const double c2 = -1.0 / 5.0;
    const double c3 = 8.0 / 315.0;
    const double c4 = -1.0 / 560.0;

    // ����� ������� ����������
    auto start = std::chrono::high_resolution_clock::now();

    // �������� ���� �� �������
    for (int n = 0; n < Nt; ++n) {
        double t = n * dt;  // ������� �����

        // ���������� �������� ������
        double arg = M_PI * f0 * (t - t0);
        double S = (1 - 2 * arg * arg) * std::exp(-arg * arg);

        // ���������� ��������� ���� �� ���������� �����
        for (int i = 5; i <= Nx-4; ++i) {
            for (int j = 5; j <= Nz-4; ++j) {
                // ������ ����������� �� x
                double d2p_dx2 = (
                    c4 * (p_n[i - 4][j] + p_n[i + 4][j]) +
                    c3 * (p_n[i - 3][j] + p_n[i + 3][j]) +
                    c2 * (p_n[i - 2][j] + p_n[i + 2][j]) +
                    c1 * (p_n[i - 1][j] + p_n[i + 1][j]) +
                    c0 * p_n[i][j]
                    ) / (dx * dx);

                // ������ ����������� �� z
                double d2p_dz2 = (
                    c4 * (p_n[i][j - 4] + p_n[i][j + 4]) +
                    c3 * (p_n[i][j - 3] + p_n[i][j + 3]) +
                    c2 * (p_n[i][j - 2] + p_n[i][j + 2]) +
                    c1 * (p_n[i][j - 1] + p_n[i][j + 1]) +
                    c0 * p_n[i][j]
                    ) / (dz * dz);

                // ���������� �������� � ��������� ��������� ����
                p_np1[i][j] = 2 * p_n[i][j] - p_nm1[i][j] + v * v * dt * dt * (d2p_dx2 + d2p_dz2);

                // ���������� ��������� � ����������� �����
                if (i == is && j == js) {
                    p_np1[i][j] += v * v * dt * dt * S;
                }
            }
        }

        // ���������� ��������� ������� ������� (p = 0 �� ��������)
        for (int i = 0; i <= Nx + 1; ++i) {
            p_np1[i][0] = 0.0;         // ������ �������
            p_np1[i][Nz + 1] = 0.0;    // ������� �������
        }
        for (int j = 0; j <= Nz + 1; ++j) {
            p_np1[0][j] = 0.0;         // ����� �������
            p_np1[Nx + 1][j] = 0.0;    // ������ �������
        }

        // ���������� ��������� �����
        p_nm1 = p_n;
        p_n = p_np1;
        if (n % 100 == 0) {
            std::string filename = "results_" + std::to_string(n / 100) + ".csv";
           // saveResults(p_n, filename);
        }
    }

    // ����� ������� ����������
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " second" << std::endl;
    //saveResults(p_n, "results.csv");
    std::cout << "���������� ��������� � ���� results.csv" << std::endl;
    // �������� �������� ���� ��������� � p_n
    // ����� ����� �������� ��� ��� ���������� ��� ������� �����������
}
/*
int main() {
    solveWaveEquationSequential();

    return 0;
}*/