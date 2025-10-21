// Matrix class with parallel operations
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

class Matrix {
public:
    int rows, cols;
    vector<double> data;

    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

    double& operator()(int i, int j) {
        return data[i * cols + j];
    }

    const double& operator()(int i, int j) const {
        return data[i * cols + j];
    }

    void fill_random() {
        #pragma omp parallel for
        for (int i = 0; i < rows * cols; i++) {
            data[i] = (double)rand() / RAND_MAX;
        }
    }

    Matrix operator*(const Matrix& other) const {
        Matrix result(rows, other.cols);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < cols; k++) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    Matrix operator+(const Matrix& other) const {
        Matrix result(rows, cols);
        #pragma omp parallel for
        for (int i = 0; i < rows * cols; i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    Matrix transpose() const {
        Matrix result(cols, rows);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    void print(int max_display = 5) const {
        int display_rows = min(rows, max_display);
        int display_cols = min(cols, max_display);

        for (int i = 0; i < display_rows; i++) {
            for (int j = 0; j < display_cols; j++) {
                cout << (*this)(i, j) << " ";
            }
            if (cols > max_display) cout << "...";
            cout << endl;
        }
        if (rows > max_display) cout << "..." << endl;
    }
};

int main() {
    const int N = 512;

    cout << "Matrix operations (" << N << "x" << N << ")" << endl;
    cout << "Threads: " << omp_get_max_threads() << endl << endl;

    Matrix A(N, N), B(N, N);
    A.fill_random();
    B.fill_random();

    double start = omp_get_wtime();
    Matrix C = A * B;
    double time = omp_get_wtime() - start;
    cout << "Matrix multiplication: " << time << "s" << endl;

    start = omp_get_wtime();
    Matrix D = A + B;
    time = omp_get_wtime() - start;
    cout << "Matrix addition: " << time << "s" << endl;

    start = omp_get_wtime();
    Matrix E = A.transpose();
    time = omp_get_wtime() - start;
    cout << "Matrix transpose: " << time << "s" << endl;

    cout << "\nSample from result (C = A * B):" << endl;
    C.print();

    return 0;
}
