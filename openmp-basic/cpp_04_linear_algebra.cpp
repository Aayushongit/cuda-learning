// Linear algebra: AXPY, matrix-vector product, Jacobi iteration
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

// AXPY: y = a*x + y
void axpy(double a, const vector<double>& x, vector<double>& y) {
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++) {
        y[i] = a * x[i] + y[i];
    }
}

// Matrix-vector product: y = A*x
void matvec(const vector<vector<double>>& A, const vector<double>& x, vector<double>& y) {
    int n = A.size();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
}

// Jacobi iteration for solving Ax = b
vector<double> jacobi_solver(const vector<vector<double>>& A,
                             const vector<double>& b,
                             int max_iter, double tol) {
    int n = A.size();
    vector<double> x(n, 0.0);
    vector<double> x_new(n);

    for (int iter = 0; iter < max_iter; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double sigma = 0.0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sigma += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sigma) / A[i][i];
        }

        // Check convergence
        double error = 0.0;
        #pragma omp parallel for reduction(max:error)
        for (int i = 0; i < n; i++) {
            error = max(error, abs(x_new[i] - x[i]));
        }

        x = x_new;

        if (error < tol) {
            cout << "Converged in " << iter + 1 << " iterations" << endl;
            break;
        }
    }

    return x;
}

int main() {
    const int N = 1000;

    // Test AXPY
    vector<double> x(N), y(N);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        x[i] = i * 0.5;
        y[i] = i * 0.3;
    }

    cout << "AXPY operation (y = 2.5*x + y)" << endl;
    double start = omp_get_wtime();
    axpy(2.5, x, y);
    cout << "Time: " << (omp_get_wtime() - start) << "s" << endl << endl;

    // Test Matrix-Vector product
    int n = 500;
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> vec(n), result(n);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (i == j) ? 2.0 : -0.5;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        vec[i] = 1.0;
    }

    cout << "Matrix-vector product (" << n << "x" << n << ")" << endl;
    start = omp_get_wtime();
    matvec(A, vec, result);
    cout << "Time: " << (omp_get_wtime() - start) << "s" << endl << endl;

    // Jacobi solver for diagonally dominant system
    n = 100;
    vector<vector<double>> A_sys(n, vector<double>(n, 0.0));
    vector<double> b_sys(n);

    // Create diagonally dominant system
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        A_sys[i][i] = 4.0;
        if (i > 0) A_sys[i][i-1] = -1.0;
        if (i < n-1) A_sys[i][i+1] = -1.0;
        b_sys[i] = 1.0;
    }

    cout << "Jacobi solver for " << n << "x" << n << " system" << endl;
    start = omp_get_wtime();
    vector<double> solution = jacobi_solver(A_sys, b_sys, 1000, 1e-6);
    cout << "Time: " << (omp_get_wtime() - start) << "s" << endl;
    cout << "First 5 solution values: ";
    for (int i = 0; i < 5; i++) {
        cout << solution[i] << " ";
    }
    cout << endl;

    return 0;
}
