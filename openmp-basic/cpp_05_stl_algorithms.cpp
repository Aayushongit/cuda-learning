// Parallel STL-style algorithms
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>

using namespace std;

// Parallel transform
template<typename T, typename UnaryOp>
void parallel_transform(vector<T>& v, UnaryOp op) {
    #pragma omp parallel for
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = op(v[i]);
    }
}

// Parallel reduce
template<typename T>
T parallel_reduce(const vector<T>& v, T init) {
    T result = init;
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < v.size(); i++) {
        result += v[i];
    }
    return result;
}

// Parallel find_if
template<typename T, typename Predicate>
int parallel_find_if(const vector<T>& v, Predicate pred) {
    int result = -1;

    #pragma omp parallel for
    for (size_t i = 0; i < v.size(); i++) {
        if (pred(v[i]) && result == -1) {
            #pragma omp critical
            {
                if (result == -1) {
                    result = i;
                }
            }
        }
    }
    return result;
}

// Parallel count_if
template<typename T, typename Predicate>
int parallel_count_if(const vector<T>& v, Predicate pred) {
    int count = 0;
    #pragma omp parallel for reduction(+:count)
    for (size_t i = 0; i < v.size(); i++) {
        if (pred(v[i])) {
            count++;
        }
    }
    return count;
}

// Parallel partial sum (prefix sum)
template<typename T>
vector<T> parallel_prefix_sum(const vector<T>& v) {
    int n = v.size();
    vector<T> result(n);

    // Simple parallel approach (not optimal but demonstrates concept)
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            T sum = 0;
            for (int j = 0; j <= i; j++) {
                sum += v[j];
            }
            result[i] = sum;
        }
    }

    return result;
}

int main() {
    const size_t N = 10000000;

    vector<double> data(N);

    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        data[i] = i % 1000;
    }

    cout << "Parallel STL-style algorithms" << endl;
    cout << "Data size: " << N << endl;
    cout << "Threads: " << omp_get_max_threads() << endl << endl;

    // Transform: square all elements
    cout << "Transform (square all elements)" << endl;
    double start = omp_get_wtime();
    parallel_transform(data, [](double x) { return x * x; });
    cout << "Time: " << (omp_get_wtime() - start) << "s" << endl << endl;

    // Reduce: sum all elements
    cout << "Reduce (sum all elements)" << endl;
    start = omp_get_wtime();
    double sum = parallel_reduce(data, 0.0);
    cout << "Sum: " << sum << " (" << (omp_get_wtime() - start) << "s)" << endl << endl;

    // Count_if: count elements > 500000
    cout << "Count_if (count elements > 500000)" << endl;
    start = omp_get_wtime();
    int count = parallel_count_if(data, [](double x) { return x > 500000; });
    cout << "Count: " << count << " (" << (omp_get_wtime() - start) << "s)" << endl << endl;

    // Find_if: find first element > 1000000
    cout << "Find_if (find first > 1000000)" << endl;
    start = omp_get_wtime();
    int idx = parallel_find_if(data, [](double x) { return x > 1000000; });
    cout << "Index: " << idx << " (" << (omp_get_wtime() - start) << "s)" << endl << endl;

    // Prefix sum on smaller dataset
    vector<int> small_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    cout << "Prefix sum [1,2,3,4,5,6,7,8,9,10]:" << endl;
    vector<int> prefix = parallel_prefix_sum(small_data);
    cout << "Result: ";
    for (int x : prefix) cout << x << " ";
    cout << endl;

    return 0;
}
