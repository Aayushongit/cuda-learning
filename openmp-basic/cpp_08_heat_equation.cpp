// Heat equation solver using finite differences (2D)
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

class HeatSolver2D {
private:
    int nx, ny;
    double dx, dy, dt, alpha;
    vector<vector<double>> u, u_new;

public:
    HeatSolver2D(int nx, int ny, double Lx, double Ly, double alpha, double dt)
        : nx(nx), ny(ny), alpha(alpha), dt(dt),
          dx(Lx / (nx - 1)), dy(Ly / (ny - 1)),
          u(nx, vector<double>(ny, 0.0)),
          u_new(nx, vector<double>(ny, 0.0)) {}

    void set_initial_condition(function<double(double, double)> f) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                double x = i * dx;
                double y = j * dy;
                u[i][j] = f(x, y);
            }
        }
    }

    void set_boundary_conditions() {
        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            u[i][0] = 0.0;
            u[i][ny-1] = 0.0;
        }

        #pragma omp parallel for
        for (int j = 0; j < ny; j++) {
            u[0][j] = 0.0;
            u[nx-1][j] = 0.0;
        }
    }

    void step() {
        double rx = alpha * dt / (dx * dx);
        double ry = alpha * dt / (dy * dy);

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                u_new[i][j] = u[i][j] +
                    rx * (u[i+1][j] - 2*u[i][j] + u[i-1][j]) +
                    ry * (u[i][j+1] - 2*u[i][j] + u[i][j-1]);
            }
        }

        swap(u, u_new);
        set_boundary_conditions();
    }

    void solve(int num_steps) {
        for (int step = 0; step < num_steps; step++) {
            this->step();
        }
    }

    double get_max_temperature() const {
        double max_temp = 0.0;
        #pragma omp parallel for collapse(2) reduction(max:max_temp)
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                if (u[i][j] > max_temp) {
                    max_temp = u[i][j];
                }
            }
        }
        return max_temp;
    }

    double get_total_heat() const {
        double total = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:total)
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                total += u[i][j];
            }
        }
        return total * dx * dy;
    }

    void print_center_region(int size = 5) const {
        int cx = nx / 2;
        int cy = ny / 2;

        cout << "Center region temperatures:" << endl;
        for (int i = cx - size/2; i <= cx + size/2; i++) {
            for (int j = cy - size/2; j <= cy + size/2; j++) {
                printf("%.3f ", u[i][j]);
            }
            cout << endl;
        }
    }
};

int main() {
    const int NX = 256;
    const int NY = 256;
    const double LX = 1.0;
    const double LY = 1.0;
    const double ALPHA = 0.01;
    const double DT = 0.0001;
    const int STEPS = 1000;

    cout << "2D Heat Equation Solver" << endl;
    cout << "Grid: " << NX << "x" << NY << endl;
    cout << "Steps: " << STEPS << endl;
    cout << "Threads: " << omp_get_max_threads() << endl << endl;

    HeatSolver2D solver(NX, NY, LX, LY, ALPHA, DT);

    // Initial condition: hot spot in the center
    auto initial = [](double x, double y) {
        double dx = x - 0.5;
        double dy = y - 0.5;
        double r2 = dx*dx + dy*dy;
        return 100.0 * exp(-r2 / 0.01);
    };

    solver.set_initial_condition(initial);
    solver.set_boundary_conditions();

    cout << "Initial max temperature: " << solver.get_max_temperature() << endl;
    cout << "Initial total heat: " << solver.get_total_heat() << endl << endl;

    double start = omp_get_wtime();
    solver.solve(STEPS);
    double elapsed = omp_get_wtime() - start;

    cout << "Simulation time: " << elapsed << "s" << endl;
    cout << "Final max temperature: " << solver.get_max_temperature() << endl;
    cout << "Final total heat: " << solver.get_total_heat() << endl << endl;

    solver.print_center_region(5);

    return 0;
}
