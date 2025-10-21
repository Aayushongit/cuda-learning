// Vector operations: dot product, norm, element-wise operations
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

double dot_product(const vector<double>& a, const vector<double>& b) {
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

double vector_norm(const vector<double>& v) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < v.size(); i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

vector<double> vector_add(const vector<double>& a, const vector<double>& b) {
    vector<double> result(a.size());
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

vector<double> scalar_multiply(const vector<double>& v, double scalar) {
    vector<double> result(v.size());
    #pragma omp parallel for
    for (size_t i = 0; i < v.size(); i++) {
        result[i] = v[i] * scalar;
    }
    return result;
}

int main() {
    const size_t N = 10000000;

    vector<double> a(N), b(N);

    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        a[i] = i * 0.5;
        b[i] = i * 0.3;
    }

    cout << "Vector size: " << N << endl;
    cout << "Threads: " << omp_get_max_threads() << endl << endl;

    double start = omp_get_wtime();
    double dot = dot_product(a, b);
    double time = omp_get_wtime() - start;
    cout << "Dot product: " << dot << " (" << time << "s)" << endl;

    start = omp_get_wtime();
    double norm = vector_norm(a);
    time = omp_get_wtime() - start;
    cout << "Norm of a: " << norm << " (" << time << "s)" << endl;

    start = omp_get_wtime();
    vector<double> c = vector_add(a, b);
    time = omp_get_wtime() - start;
    cout << "Vector addition done (" << time << "s)" << endl;

    return 0;
}
