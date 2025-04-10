#pragma once
#include <fstream>
#include <vector>
#define M_PI 3.14159265
// Допустим, поле результатов хранится в контейнере p_n, размер которого равен (Nx+2) x (Nz+2)
void saveResults(const std::vector<std::vector<double>>& p_n, const std::string& filename) {
    std::ofstream file(filename);
    for (size_t i = 0; i < p_n.size(); i++) {
        for (size_t j = 0; j < p_n[i].size(); j++) {
            file << p_n[i][j];
            if (j < p_n[i].size() - 1)
                file << ",";
        }
        file << "\n";
    }
    file.close();
}