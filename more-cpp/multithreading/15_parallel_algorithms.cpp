// Parallel Algorithms (C++17)
// Standard algorithms can run in parallel using execution policies

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <chrono>
#include <random>

// Helper function to measure execution time
template<typename Func>
void measureTime(const std::string& name, Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << name << ": " << duration.count() << "ms" << std::endl;
}

// Expensive computation for demonstration
int expensiveComputation(int x) {
    int result = x;
    for (int i = 0; i < 10000; i++) {
        result = (result * 13 + 7) % 1000000;
    }
    return result;
}

int main() {
    std::cout << "=== Parallel Algorithms (C++17) ===" << std::endl;

    // Create large vector
    const size_t SIZE = 1000000;
    std::vector<int> data(SIZE);

    // Fill with random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000);

    for (auto& val : data) {
        val = dis(gen);
    }

    std::cout << "\n=== std::sort ===" << std::endl;

    // Sequential sort
    auto data1 = data;
    measureTime("Sequential sort", [&]() {
        std::sort(data1.begin(), data1.end());
    });

    // Parallel sort
    auto data2 = data;
    measureTime("Parallel sort", [&]() {
        std::sort(std::execution::par, data2.begin(), data2.end());
    });

    std::cout << "\n=== std::for_each ===" << std::endl;

    std::vector<int> small_data(1000);
    std::iota(small_data.begin(), small_data.end(), 1);

    // Sequential for_each
    auto small1 = small_data;
    measureTime("Sequential for_each", [&]() {
        std::for_each(small1.begin(), small1.end(), [](int& x) {
            x = expensiveComputation(x);
        });
    });

    // Parallel for_each
    auto small2 = small_data;
    measureTime("Parallel for_each", [&]() {
        std::for_each(std::execution::par, small2.begin(), small2.end(), [](int& x) {
            x = expensiveComputation(x);
        });
    });

    std::cout << "\n=== std::transform ===" << std::endl;

    std::vector<int> input(100000);
    std::iota(input.begin(), input.end(), 1);
    std::vector<int> output(input.size());

    // Sequential transform
    measureTime("Sequential transform", [&]() {
        std::transform(input.begin(), input.end(), output.begin(),
                      [](int x) { return x * x; });
    });

    // Parallel transform
    measureTime("Parallel transform", [&]() {
        std::transform(std::execution::par, input.begin(), input.end(), output.begin(),
                      [](int x) { return x * x; });
    });

    std::cout << "\n=== std::reduce ===" << std::endl;

    std::vector<int> nums(10000000);
    std::iota(nums.begin(), nums.end(), 1);

    // Sequential reduce
    long long sum1;
    measureTime("Sequential reduce", [&]() {
        sum1 = std::reduce(nums.begin(), nums.end());
    });

    // Parallel reduce
    long long sum2;
    measureTime("Parallel reduce", [&]() {
        sum2 = std::reduce(std::execution::par, nums.begin(), nums.end());
    });

    std::cout << "Sum (sequential): " << sum1 << std::endl;
    std::cout << "Sum (parallel): " << sum2 << std::endl;

    std::cout << "\n=== Execution Policies ===" << std::endl;
    std::cout << "std::execution::seq - Sequential execution" << std::endl;
    std::cout << "std::execution::par - Parallel execution" << std::endl;
    std::cout << "std::execution::par_unseq - Parallel + vectorized" << std::endl;

    return 0;
}
