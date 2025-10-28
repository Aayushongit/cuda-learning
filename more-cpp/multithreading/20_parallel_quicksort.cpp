// Parallel Quicksort
// Another divide-and-conquer algorithm using multithreading

#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <random>

// Partition function for quicksort
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }

    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Sequential quicksort
void sequentialQuicksort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot = partition(arr, low, high);

        sequentialQuicksort(arr, low, pivot - 1);
        sequentialQuicksort(arr, pivot + 1, high);
    }
}

// Parallel quicksort with depth limit
void parallelQuicksort(std::vector<int>& arr, int low, int high, int depth) {
    if (low >= high) return;

    int pivot = partition(arr, low, high);

    // Create threads only up to certain depth
    if (depth > 0) {
        std::thread leftThread(parallelQuicksort, std::ref(arr), low, pivot - 1, depth - 1);
        std::thread rightThread(parallelQuicksort, std::ref(arr), pivot + 1, high, depth - 1);

        leftThread.join();
        rightThread.join();
    } else {
        // Sequential for small partitions
        sequentialQuicksort(arr, low, pivot - 1);
        sequentialQuicksort(arr, pivot + 1, high);
    }
}

// Wrapper
void parallelQuicksort(std::vector<int>& arr) {
    if (arr.empty()) return;
    int maxDepth = std::thread::hardware_concurrency();
    parallelQuicksort(arr, 0, arr.size() - 1, maxDepth);
}

// Print first n elements
void printArray(const std::vector<int>& arr, int n = 20) {
    std::cout << "[";
    for (int i = 0; i < std::min(n, (int)arr.size()); i++) {
        std::cout << arr[i];
        if (i < n - 1 && i < (int)arr.size() - 1) std::cout << ", ";
    }
    if ((int)arr.size() > n) std::cout << "...";
    std::cout << "]" << std::endl;
}

// Verify sorted
bool isSorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) return false;
    }
    return true;
}

// Measure execution time
template<typename Func>
long long measureTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main() {
    std::cout << "=== Parallel Quicksort ===" << std::endl;

    const int SIZE = 1000000;

    // Generate random data
    std::vector<int> data(SIZE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10000);

    for (auto& val : data) {
        val = dis(gen);
    }

    std::cout << "Array size: " << SIZE << std::endl;
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;

    std::cout << "\nBefore sorting: ";
    printArray(data, 10);

    // Sequential quicksort
    auto data1 = data;
    long long time1 = measureTime([&]() {
        sequentialQuicksort(data1, 0, data1.size() - 1);
    });

    std::cout << "\n--- Sequential Quicksort ---" << std::endl;
    std::cout << "Time: " << time1 << "ms" << std::endl;
    std::cout << "Sorted correctly: " << (isSorted(data1) ? "YES" : "NO") << std::endl;
    std::cout << "After sorting: ";
    printArray(data1, 10);

    // Parallel quicksort
    auto data2 = data;
    long long time2 = measureTime([&]() {
        parallelQuicksort(data2);
    });

    std::cout << "\n--- Parallel Quicksort ---" << std::endl;
    std::cout << "Time: " << time2 << "ms" << std::endl;
    std::cout << "Sorted correctly: " << (isSorted(data2) ? "YES" : "NO") << std::endl;
    std::cout << "After sorting: ";
    printArray(data2, 10);

    // STL sort for comparison
    auto data3 = data;
    long long time3 = measureTime([&]() {
        std::sort(data3.begin(), data3.end());
    });

    std::cout << "\n--- STL sort ---" << std::endl;
    std::cout << "Time: " << time3 << "ms" << std::endl;

    // Performance comparison
    std::cout << "\n--- Performance Comparison ---" << std::endl;
    std::cout << "Sequential: " << time1 << "ms" << std::endl;
    std::cout << "Parallel: " << time2 << "ms" << std::endl;
    std::cout << "STL sort: " << time3 << "ms" << std::endl;

    double speedup = (double)time1 / time2;
    std::cout << "\nParallel speedup: " << speedup << "x" << std::endl;

    if (time2 < time1) {
        double improvement = ((double)(time1 - time2) / time1) * 100;
        std::cout << "Parallel is " << improvement << "% faster than sequential" << std::endl;
    }

    return 0;
}
