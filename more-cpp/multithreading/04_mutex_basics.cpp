// Mutex Basics - Solving Race Conditions
// Demonstrates how to use std::mutex to protect shared data

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

// Shared resources
int safeCounter = 0;
std::mutex counterMutex;  // Mutex to protect the counter

// Thread-safe increment using mutex
void incrementSafe(int iterations) {
    for (int i = 0; i < iterations; i++) {
        // Lock the mutex before accessing shared data
        counterMutex.lock();
        safeCounter++;
        counterMutex.unlock();  // Always unlock after use
    }
}

// Better approach: using lock_guard (RAII - automatic unlock)
void incrementWithLockGuard(int iterations) {
    for (int i = 0; i < iterations; i++) {
        // lock_guard locks on construction, unlocks on destruction
        std::lock_guard<std::mutex> lock(counterMutex);
        safeCounter++;
        // Automatic unlock when lock goes out of scope
    }
}

int main() {
    std::cout << "=== Mutex Protection Demo ===" << std::endl;

    const int NUM_THREADS = 10;
    const int ITERATIONS = 10000;

    std::vector<std::thread> threads;

    // Create threads with mutex protection
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.push_back(std::thread(incrementWithLockGuard, ITERATIONS));
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    int expected = NUM_THREADS * ITERATIONS;
    std::cout << "Expected counter value: " << expected << std::endl;
    std::cout << "Actual counter value: " << safeCounter << std::endl;

    if (safeCounter == expected) {
        std::cout << "\nSUCCESS! Mutex prevented race condition" << std::endl;
    } else {
        std::cout << "\nUnexpected result!" << std::endl;
    }

    return 0;
}
