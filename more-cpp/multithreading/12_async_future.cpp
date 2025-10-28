// Async and Future - Task-based Parallelism
// std::async launches async tasks, std::future gets results

#include <iostream>
#include <future>
#include <chrono>
#include <thread>
#include <vector>

// Simple function that returns a value
int calculate(int x, int y) {
    std::cout << "Calculating " << x << " + " << y << "..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return x + y;
}

// Function that might throw exception
int riskyCalculation(int x) {
    if (x < 0) {
        throw std::runtime_error("Negative input not allowed");
    }
    return x * x;
}

// Long running task
long long fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    std::cout << "=== std::async and std::future ===" << std::endl;

    // Launch async task (may run in new thread or deferred)
    std::future<int> result1 = std::async(calculate, 10, 20);

    std::cout << "Main thread doing other work..." << std::endl;

    // Get result (blocks until ready)
    int sum = result1.get();
    std::cout << "Result: " << sum << std::endl;

    std::cout << "\n=== Launch Policies ===" << std::endl;

    // std::launch::async - definitely runs in new thread
    auto future2 = std::async(std::launch::async, calculate, 5, 15);

    // std::launch::deferred - runs when get() is called (lazy evaluation)
    auto future3 = std::async(std::launch::deferred, calculate, 3, 7);

    std::cout << "Getting deferred result..." << std::endl;
    std::cout << "Deferred result: " << future3.get() << std::endl;
    std::cout << "Async result: " << future2.get() << std::endl;

    std::cout << "\n=== Exception Handling ===" << std::endl;

    auto future4 = std::async(riskyCalculation, -5);

    try {
        int result = future4.get();
        std::cout << "Result: " << result << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }

    std::cout << "\n=== Multiple Async Tasks ===" << std::endl;

    // Launch multiple calculations in parallel
    std::vector<std::future<long long>> futures;
    std::vector<int> inputs = {35, 36, 37, 38};

    for (int n : inputs) {
        futures.push_back(std::async(std::launch::async, fibonacci, n));
    }

    // Collect results
    for (size_t i = 0; i < futures.size(); i++) {
        std::cout << "fib(" << inputs[i] << ") = " << futures[i].get() << std::endl;
    }

    std::cout << "\n=== Checking Future Status ===" << std::endl;

    auto future5 = std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        return 42;
    });

    // Check if result is ready
    while (future5.wait_for(std::chrono::milliseconds(500)) != std::future_status::ready) {
        std::cout << "Still waiting..." << std::endl;
    }

    std::cout << "Result ready: " << future5.get() << std::endl;

    return 0;
}
