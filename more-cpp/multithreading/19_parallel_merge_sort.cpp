// Parallel Merge Sort
// Demonstrates divide-and-conquer algorithm with multithreading

#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <random>

// Merge two sorted subarrays
void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    // Merge in sorted order
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    // Copy remaining elements
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    // Copy back to original array
    for (int i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }
}

// Sequential merge sort
void sequentialMergeSort(std::vector<int>& arr, int left, int right) {
    if (left >= right) return;

    int mid = left + (right - left) / 2;

    sequentialMergeSort(arr, left, mid);
    sequentialMergeSort(arr, mid + 1, right);

    merge(arr, left, mid, right);
}

// Parallel merge sort with depth limit
void parallelMergeSort(std::vector<int>& arr, int left, int right, int depth) {
    if (left >= right) return;

    int mid = left + (right - left) / 2;

    // Use threads only up to certain depth to avoid too many threads
    if (depth > 0) {
        // Sort left and right halves in parallel
        std::thread leftThread(parallelMergeSort, std::ref(arr), left, mid, depth - 1);
        std::thread rightThread(parallelMergeSort, std::ref(arr), mid + 1, right, depth - 1);

        leftThread.join();
        rightThread.join();
    } else {
        // Fall back to sequential for small subarrays
        sequentialMergeSort(arr, left, mid);
        sequentialMergeSort(arr, mid + 1, right);
    }

    merge(arr, left, mid, right);
}

// Wrapper function
void parallelMergeSort(std::vector<int>& arr) {
    int maxDepth = std::thread::hardware_concurrency();
    parallelMergeSort(arr, 0, arr.size() - 1, maxDepth);
}

// Utility: print first n elements
void printArray(const std::vector<int>& arr, int n = 20) {
    std::cout << "[";
    for (int i = 0; i < std::min(n, (int)arr.size()); i++) {
        std::cout << arr[i];
        if (i < n - 1 && i < (int)arr.size() - 1) std::cout << ", ";
    }
    if ((int)arr.size() > n) std::cout << "...";
    std::cout << "]" << std::endl;
}

// Verify array is sorted
bool isSorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) return false;
    }
    return true;
}

int main() {
    std::cout << "=== Parallel Merge Sort ===" << std::endl;

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

    // Test sequential merge sort
    auto data1 = data;
    auto start = std::chrono::high_resolution_clock::now();
    sequentialMergeSort(data1, 0, data1.size() - 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "\n--- Sequential Merge Sort ---" << std::endl;
    std::cout << "Time: " << duration1.count() << "ms" << std::endl;
    std::cout << "Sorted correctly: " << (isSorted(data1) ? "YES" : "NO") << std::endl;
    std::cout << "After sorting: ";
    printArray(data1, 10);

    // Test parallel merge sort
    auto data2 = data;
    start = std::chrono::high_resolution_clock::now();
    parallelMergeSort(data2);
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "\n--- Parallel Merge Sort ---" << std::endl;
    std::cout << "Time: " << duration2.count() << "ms" << std::endl;
    std::cout << "Sorted correctly: " << (isSorted(data2) ? "YES" : "NO") << std::endl;
    std::cout << "After sorting: ";
    printArray(data2, 10);

    // Compare
    std::cout << "\n--- Performance Comparison ---" << std::endl;
    double speedup = (double)duration1.count() / duration2.count();
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    if (duration2.count() < duration1.count()) {
        double improvement = ((double)(duration1.count() - duration2.count()) / duration1.count()) * 100;
        std::cout << "Parallel is " << improvement << "% faster" << std::endl;
    }

    return 0;
}
