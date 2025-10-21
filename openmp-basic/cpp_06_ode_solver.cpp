// ODE solvers: Euler and RK4 methods (parallel for multiple initial conditions)
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

// dy/dt = f(t, y)
using ODEFunction = function<double(double, double)>;

// Single trajectory with RK4
vector<double> rk4_single(ODEFunction f, double y0, double t0, double tf, int steps) {
    double h = (tf - t0) / steps;
    vector<double> y(steps + 1);
    y[0] = y0;

    for (int i = 0; i < steps; i++) {
        double t = t0 + i * h;
        double k1 = f(t, y[i]);
        double k2 = f(t + h/2, y[i] + h*k1/2);
        double k3 = f(t + h/2, y[i] + h*k2/2);
        double k4 = f(t + h, y[i] + h*k3);

        y[i+1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0;
    }

    return y;
}

// Parallel simulation of multiple initial conditions
vector<vector<double>> solve_multiple_ics(ODEFunction f,
                                          const vector<double>& y0_values,
                                          double t0, double tf, int steps) {
    int n_ics = y0_values.size();
    vector<vector<double>> results(n_ics);

    #pragma omp parallel for
    for (int i = 0; i < n_ics; i++) {
        results[i] = rk4_single(f, y0_values[i], t0, tf, steps);
    }

    return results;
}

// 2D system: coupled ODEs solved in parallel
struct State2D {
    double x, y;
};

State2D rk4_2d(function<double(double, State2D)> fx,
               function<double(double, State2D)> fy,
               State2D s0, double t0, double tf, int steps) {
    double h = (tf - t0) / steps;
    State2D s = s0;

    for (int i = 0; i < steps; i++) {
        double t = t0 + i * h;

        double kx1 = fx(t, s);
        double ky1 = fy(t, s);

        State2D s2 = {s.x + h*kx1/2, s.y + h*ky1/2};
        double kx2 = fx(t + h/2, s2);
        double ky2 = fy(t + h/2, s2);

        State2D s3 = {s.x + h*kx2/2, s.y + h*ky2/2};
        double kx3 = fx(t + h/2, s3);
        double ky3 = fy(t + h/2, s3);

        State2D s4 = {s.x + h*kx3, s.y + h*ky3};
        double kx4 = fx(t + h, s4);
        double ky4 = fy(t + h, s4);

        s.x += h * (kx1 + 2*kx2 + 2*kx3 + kx4) / 6.0;
        s.y += h * (ky1 + 2*ky2 + 2*ky3 + ky4) / 6.0;
    }

    return s;
}

int main() {
    cout << "ODE Solvers with OpenMP" << endl;
    cout << "Threads: " << omp_get_max_threads() << endl << endl;

    // Example 1: Exponential growth dy/dt = k*y
    auto exponential = [](double t, double y) { return 0.5 * y; };

    vector<double> initial_conditions;
    for (int i = 1; i <= 100; i++) {
        initial_conditions.push_back(i * 0.1);
    }

    cout << "Solving " << initial_conditions.size() << " initial conditions" << endl;
    double start = omp_get_wtime();
    auto results = solve_multiple_ics(exponential, initial_conditions, 0, 10, 10000);
    double elapsed = omp_get_wtime() - start;
    cout << "Time: " << elapsed << "s" << endl;
    cout << "Sample result (y0=1.0): y(10) = " << results[9].back() << endl;
    cout << "Expected (e^5): " << exp(5) << endl << endl;

    // Example 2: Harmonic oscillator (2D system)
    // dx/dt = v, dv/dt = -omega^2 * x
    double omega = 2.0;
    auto fx = [](double t, State2D s) { return s.y; };
    auto fy = [omega](double t, State2D s) { return -omega*omega * s.x; };

    vector<State2D> initial_states;
    for (int i = 0; i < 50; i++) {
        initial_states.push_back({(double)i * 0.1, 0.0});
    }

    cout << "Harmonic oscillator (" << initial_states.size() << " trajectories)" << endl;
    start = omp_get_wtime();

    vector<State2D> final_states(initial_states.size());
    #pragma omp parallel for
    for (size_t i = 0; i < initial_states.size(); i++) {
        final_states[i] = rk4_2d(fx, fy, initial_states[i], 0, 10, 10000);
    }

    elapsed = omp_get_wtime() - start;
    cout << "Time: " << elapsed << "s" << endl;
    cout << "Sample (x0=1.0, v0=0): x(10)=" << final_states[10].x
         << ", v(10)=" << final_states[10].y << endl;

    return 0;
}
