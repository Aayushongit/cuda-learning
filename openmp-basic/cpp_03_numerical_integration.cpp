// Numerical integration: Trapezoidal and Simpson's rule
#include <iostream>
#include <cmath>
#include <omp.h>
#include <functional>

using namespace std;

double trapezoidal_serial(function<double(double)> f, double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));

    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }
    return sum * h;
}

double trapezoidal_parallel(function<double(double)> f, double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));

    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }
    return sum * h;
}

double simpson_parallel(function<double(double)> f, double a, double b, int n) {
    if (n % 2 != 0) n++;  // Simpson needs even intervals

    double h = (b - a) / n;
    double sum = f(a) + f(b);

    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        sum += (i % 2 == 0 ? 2.0 : 4.0) * f(x);
    }
    return sum * h / 3.0;
}

int main() {
    // Test functions
    auto f1 = [](double x) { return x * x; };  // Integral from 0 to 1 = 1/3
    auto f2 = [](double x) { return sin(x); }; // Integral from 0 to pi = 2
    auto f3 = [](double x) { return exp(-x*x); }; // Gaussian

    const int n = 100000000;

    cout << "Numerical Integration" << endl;
    cout << "Intervals: " << n << endl;
    cout << "Threads: " << omp_get_max_threads() << endl << endl;

    // f(x) = x^2 from 0 to 1
    cout << "Function: f(x) = x^2, [0, 1]" << endl;

    double start = omp_get_wtime();
    double result_serial = trapezoidal_serial(f1, 0, 1, n);
    double time_serial = omp_get_wtime() - start;
    cout << "Serial:   " << result_serial << " (" << time_serial << "s)" << endl;

    start = omp_get_wtime();
    double result_parallel = trapezoidal_parallel(f1, 0, 1, n);
    double time_parallel = omp_get_wtime() - start;
    cout << "Parallel: " << result_parallel << " (" << time_parallel << "s)" << endl;
    cout << "Speedup: " << time_serial/time_parallel << "x" << endl;
    cout << "Expected: 0.333333" << endl << endl;

    // f(x) = sin(x) from 0 to pi
    cout << "Function: f(x) = sin(x), [0, π]" << endl;
    result_parallel = simpson_parallel(f2, 0, M_PI, n);
    cout << "Result: " << result_parallel << " (Expected: 2.0)" << endl << endl;

    // Gaussian integral
    cout << "Function: f(x) = e^(-x^2), [0, 3]" << endl;
    result_parallel = simpson_parallel(f3, 0, 3, n);
    cout << "Result: " << result_parallel << " (Expected: ~0.886)" << endl;

    return 0;
}
