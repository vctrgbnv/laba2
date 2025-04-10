#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#define M_PI 3.14159265
#include <fstream>
#include <string>
#include"SaveData.h"

void solveWaveEquationSequential() {
    // Параметры задачи
    const double v = 3000.0;       // Скорость волны, м/с
    const double Lx = 3000.0;      // Размер области по x, м
    const double Lz = 3000.0;      // Размер области по z, м
    const double f0 = 15.0;        // Доминирующая частота импульса, Гц
    const double t_max = 1.0;      // Максимальное время моделирования, с
    const double dx = 10.0;        // Шаг сетки по x, м
    const double dz = 10.0;        // Шаг сетки по z, м
    const double dt = 0.001;       // Шаг по времени, с

    // Вычисление количества точек и шагов
    const int Nx = static_cast<int>(Lx / dx) - 1;  // Количество внутренних точек по x
    const int Nz = static_cast<int>(Lz / dz) - 1;  // Количество внутренних точек по z
    const int Nt = static_cast<int>(t_max / dt);   // Количество временных шагов
    const int is = Nx / 2;         // Индекс источника по x (в центре)
    const int js = Nz / 2;         // Индекс источника по z (в центре)
    const double t0 = 2.0 / f0;    // Временной сдвиг для импульса Рикера

    // Инициализация массивов для волнового поля (с учетом границ)
    std::vector<std::vector<double>> p_nm1(Nx + 2, std::vector<double>(Nz + 2, 0.0));  // p^{n-1}
    std::vector<std::vector<double>> p_n(Nx + 2, std::vector<double>(Nz + 2, 0.0));    // p^{n}
    std::vector<std::vector<double>> p_np1(Nx + 2, std::vector<double>(Nz + 2, 0.0));  // p^{n+1}

    // Коэффициенты для аппроксимации второй производной восьмого порядка
    const double c0 = -205.0 / 72.0;
    const double c1 = 8.0 / 5.0;
    const double c2 = -1.0 / 5.0;
    const double c3 = 8.0 / 315.0;
    const double c4 = -1.0 / 560.0;

    // Замер времени выполнения
    auto start = std::chrono::high_resolution_clock::now();

    // Основной цикл по времени
    for (int n = 0; n < Nt; ++n) {
        double t = n * dt;  // Текущее время

        // Вычисление импульса Рикера
        double arg = M_PI * f0 * (t - t0);
        double S = (1 - 2 * arg * arg) * std::exp(-arg * arg);

        // Обновление волнового поля во внутренних узлах
        for (int i = 5; i <= Nx-4; ++i) {
            for (int j = 5; j <= Nz-4; ++j) {
                // Вторая производная по x
                double d2p_dx2 = (
                    c4 * (p_n[i - 4][j] + p_n[i + 4][j]) +
                    c3 * (p_n[i - 3][j] + p_n[i + 3][j]) +
                    c2 * (p_n[i - 2][j] + p_n[i + 2][j]) +
                    c1 * (p_n[i - 1][j] + p_n[i + 1][j]) +
                    c0 * p_n[i][j]
                    ) / (dx * dx);

                // Вторая производная по z
                double d2p_dz2 = (
                    c4 * (p_n[i][j - 4] + p_n[i][j + 4]) +
                    c3 * (p_n[i][j - 3] + p_n[i][j + 3]) +
                    c2 * (p_n[i][j - 2] + p_n[i][j + 2]) +
                    c1 * (p_n[i][j - 1] + p_n[i][j + 1]) +
                    c0 * p_n[i][j]
                    ) / (dz * dz);

                // Обновление значения в следующем временном слое
                p_np1[i][j] = 2 * p_n[i][j] - p_nm1[i][j] + v * v * dt * dt * (d2p_dx2 + d2p_dz2);

                // Добавление источника в центральной точке
                if (i == is && j == js) {
                    p_np1[i][j] += v * v * dt * dt * S;
                }
            }
        }

        // Применение граничных условий Дирихле (p = 0 на границах)
        for (int i = 0; i <= Nx + 1; ++i) {
            p_np1[i][0] = 0.0;         // Нижняя граница
            p_np1[i][Nz + 1] = 0.0;    // Верхняя граница
        }
        for (int j = 0; j <= Nz + 1; ++j) {
            p_np1[0][j] = 0.0;         // Левая граница
            p_np1[Nx + 1][j] = 0.0;    // Правая граница
        }

        // Обновление временных слоев
        p_nm1 = p_n;
        p_n = p_np1;
        if (n % 100 == 0) {
            std::string filename = "results_" + std::to_string(n / 100) + ".csv";
           // saveResults(p_n, filename);
        }
    }

    // Замер времени выполнения
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " second" << std::endl;
    //saveResults(p_n, "results.csv");
    std::cout << "Результаты сохранены в файл results.csv" << std::endl;
    // Итоговое волновое поле находится в p_n
    // Здесь можно добавить код для сохранения или анализа результатов
}
/*
int main() {
    solveWaveEquationSequential();

    return 0;
}*/