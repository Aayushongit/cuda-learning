// Statistical computations: mean, variance, histogram, correlation
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;

class Statistics {
public:
    static double mean(const vector<double>& data) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < data.size(); i++) {
            sum += data[i];
        }
        return sum / data.size();
    }

    static double variance(const vector<double>& data) {
        double m = mean(data);
        double sum_sq = 0.0;

        #pragma omp parallel for reduction(+:sum_sq)
        for (size_t i = 0; i < data.size(); i++) {
            double diff = data[i] - m;
            sum_sq += diff * diff;
        }
        return sum_sq / data.size();
    }

    static double std_dev(const vector<double>& data) {
        return sqrt(variance(data));
    }

    static pair<double, double> min_max(const vector<double>& data) {
        double min_val = data[0];
        double max_val = data[0];

        #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
        for (size_t i = 0; i < data.size(); i++) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }

        return {min_val, max_val};
    }

    static vector<int> histogram(const vector<double>& data, int bins) {
        auto [min_val, max_val] = min_max(data);
        vector<int> hist(bins, 0);
        double bin_width = (max_val - min_val) / bins;

        #pragma omp parallel
        {
            vector<int> local_hist(bins, 0);

            #pragma omp for
            for (size_t i = 0; i < data.size(); i++) {
                int bin = min((int)((data[i] - min_val) / bin_width), bins - 1);
                local_hist[bin]++;
            }

            #pragma omp critical
            {
                for (int i = 0; i < bins; i++) {
                    hist[i] += local_hist[i];
                }
            }
        }

        return hist;
    }

    static double correlation(const vector<double>& x, const vector<double>& y) {
        double mean_x = mean(x);
        double mean_y = mean(y);

        double sum_xy = 0.0, sum_xx = 0.0, sum_yy = 0.0;

        #pragma omp parallel for reduction(+:sum_xy,sum_xx,sum_yy)
        for (size_t i = 0; i < x.size(); i++) {
            double dx = x[i] - mean_x;
            double dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }

        return sum_xy / sqrt(sum_xx * sum_yy);
    }

    static double covariance(const vector<double>& x, const vector<double>& y) {
        double mean_x = mean(x);
        double mean_y = mean(y);
        double sum = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < x.size(); i++) {
            sum += (x[i] - mean_x) * (y[i] - mean_y);
        }

        return sum / x.size();
    }
};

int main() {
    const size_t N = 10000000;

    vector<double> data1(N), data2(N);

    // Generate data
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        data1[i] = sin(i * 0.001) * 50 + 100;
        data2[i] = cos(i * 0.001) * 30 + 50;
    }

    cout << "Statistical Analysis" << endl;
    cout << "Data size: " << N << endl;
    cout << "Threads: " << omp_get_max_threads() << endl << endl;

    double start = omp_get_wtime();
    double m1 = Statistics::mean(data1);
    double time = omp_get_wtime() - start;
    cout << "Mean: " << m1 << " (" << time << "s)" << endl;

    start = omp_get_wtime();
    double var1 = Statistics::variance(data1);
    time = omp_get_wtime() - start;
    cout << "Variance: " << var1 << " (" << time << "s)" << endl;

    start = omp_get_wtime();
    double std1 = Statistics::std_dev(data1);
    time = omp_get_wtime() - start;
    cout << "Std Dev: " << std1 << " (" << time << "s)" << endl;

    start = omp_get_wtime();
    auto [min_val, max_val] = Statistics::min_max(data1);
    time = omp_get_wtime() - start;
    cout << "Min: " << min_val << ", Max: " << max_val
         << " (" << time << "s)" << endl << endl;

    start = omp_get_wtime();
    double corr = Statistics::correlation(data1, data2);
    time = omp_get_wtime() - start;
    cout << "Correlation(data1, data2): " << corr << " (" << time << "s)" << endl;

    start = omp_get_wtime();
    double cov = Statistics::covariance(data1, data2);
    time = omp_get_wtime() - start;
    cout << "Covariance(data1, data2): " << cov << " (" << time << "s)" << endl << endl;

    // Histogram on smaller dataset
    vector<double> small_data(10000);
    #pragma omp parallel for
    for (int i = 0; i < 10000; i++) {
        small_data[i] = (i % 100) * 0.5;
    }

    start = omp_get_wtime();
    vector<int> hist = Statistics::histogram(small_data, 10);
    time = omp_get_wtime() - start;

    cout << "Histogram (10 bins): ";
    for (int count : hist) {
        cout << count << " ";
    }
    cout << "(" << time << "s)" << endl;

    return 0;
}
