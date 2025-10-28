// Barrier and Latch (C++20)
// Synchronization primitives for coordinating multiple threads

#include <iostream>
#include <thread>
#include <vector>
#include <barrier>
#include <latch>
#include <chrono>
#include <c++/11/latch>

// Latch: One-time synchronization point that counts down
void latchExample() {
    std::cout << "=== Latch Example ===" << std::endl;
    std::cout << "Waiting for all workers to be ready...\n" << std::endl;

    const int NUM_WORKERS = 5;
    std::latch workersReady(NUM_WORKERS);  // Count down from 5

    std::vector<std::thread> workers;

    for (int i = 1; i <= NUM_WORKERS; i++) {
        workers.emplace_back([&workersReady, i]() {
            // Simulate initialization
            std::cout << "Worker " << i << ": Initializing..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * i));

            std::cout << "Worker " << i << ": Ready!" << std::endl;
            workersReady.count_down();  // Decrement counter

            // Wait for all workers
            workersReady.wait();

            std::cout << "Worker " << i << ": Starting work!" << std::endl;
        });
    }

    // Main thread can also wait
    workersReady.wait();
    std::cout << "\nAll workers are ready and working!" << std::endl;

    for (auto& t : workers) {
        t.join();
    }
}

// Barrier: Reusable synchronization point with completion function
void barrierExample() {
    std::cout << "\n=== Barrier Example ===" << std::endl;
    std::cout << "Running 3 iterations with barrier synchronization\n" << std::endl;

    const int NUM_THREADS = 4;
    int iteration = 0;

    // Barrier with completion function (called when all threads arrive)
    std::barrier syncPoint(NUM_THREADS, [&]() noexcept {
        iteration++;
        std::cout << "\n>>> Iteration " << iteration << " complete <<<\n" << std::endl;
    });

    std::vector<std::thread> threads;

    for (int id = 1; id <= NUM_THREADS; id++) {
        threads.emplace_back([&syncPoint, id]() {
            for (int i = 0; i < 3; i++) {
                // Do work
                std::cout << "Thread " << id << ": Working on task " << (i + 1) << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100 * id));

                // Wait for all threads to complete this iteration
                std::cout << "Thread " << id << ": Waiting at barrier" << std::endl;
                syncPoint.arrive_and_wait();

                std::cout << "Thread " << id << ": Proceeding to next iteration" << std::endl;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "\nAll iterations completed!" << std::endl;
}

// Barrier with arrive_and_drop (thread leaves the barrier)
void arriveAndDropExample() {
    std::cout << "\n=== Barrier arrive_and_drop Example ===" << std::endl;

    const int NUM_THREADS = 4;
    std::barrier syncPoint(NUM_THREADS);

    std::vector<std::thread> threads;

    for (int id = 1; id <= NUM_THREADS; id++) {
        threads.emplace_back([&syncPoint, id]() {
            std::cout << "Thread " << id << ": Phase 1" << std::endl;
            syncPoint.arrive_and_wait();

            if (id <= 2) {
                // Only threads 1-2 continue to phase 2
                std::cout << "Thread " << id << ": Phase 2" << std::endl;
                syncPoint.arrive_and_wait();
            } else {
                // Threads 3-4 drop out
                std::cout << "Thread " << id << ": Dropping out" << std::endl;
                syncPoint.arrive_and_drop();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

// Practical example: parallel matrix computation
void parallelMatrixExample() {
    std::cout << "\n=== Parallel Matrix Computation ===" << std::endl;

    const int ROWS = 4;
    const int ITERATIONS = 3;

    std::barrier rowBarrier(ROWS, []() noexcept {
        std::cout << "All rows updated" << std::endl;
    });

    std::vector<std::thread> rowThreads;

    for (int row = 0; row < ROWS; row++) {
        rowThreads.emplace_back([&rowBarrier, row]() {
            for (int iter = 0; iter < ITERATIONS; iter++) {
                // Compute row values
                std::cout << "Computing row " << row << ", iteration " << iter << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));

                // Wait for all rows to finish before next iteration
                rowBarrier.arrive_and_wait();
            }
        });
    }

    for (auto& t : rowThreads) {
        t.join();
    }
}

int main() {
    latchExample();
    barrierExample();
    arriveAndDropExample();
    parallelMatrixExample();

    return 0;
}
