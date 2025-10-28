// Atomic Operations
// Lock-free thread-safe operations using std::atomic

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>

// Atomic counter (thread-safe without mutex)
std::atomic<int> atomicCounter(0);

// Regular int for comparison
int regularCounter = 0;

// Increment atomic counter
void incrementAtomic(int iterations) {
    for (int i = 0; i < iterations; i++) {
        atomicCounter++;  // This is atomic, thread-safe
        // Same as: atomicCounter.fetch_add(1);
    }
}

// Increment regular counter (unsafe)
void incrementRegular(int iterations) {
    for (int i = 0; i < iterations; i++) {
        regularCounter++;  // NOT thread-safe
    }
}

// Demonstrate atomic operations
void demonstrateAtomicOps() {
    std::atomic<int> value(10);

    std::cout << "\n=== Atomic Operations ===" << std::endl;
    std::cout << "Initial value: " << value << std::endl;

    // Fetch and add
    int oldValue = value.fetch_add(5);
    std::cout << "After fetch_add(5): old=" << oldValue << ", new=" << value << std::endl;

    // Fetch and subtract
    oldValue = value.fetch_sub(3);
    std::cout << "After fetch_sub(3): old=" << oldValue << ", new=" << value << std::endl;

    // Exchange (set new value, return old)
    oldValue = value.exchange(100);
    std::cout << "After exchange(100): old=" << oldValue << ", new=" << value << std::endl;

    // Compare and exchange (CAS - Compare And Swap)
    int expected = 100;
    bool success = value.compare_exchange_strong(expected, 200);
    std::cout << "Compare-exchange (expect 100, set 200): "
              << (success ? "SUCCESS" : "FAILED") << ", value=" << value << std::endl;
}

// Demonstrate memory ordering
void demonstrateMemoryOrder() {
    std::atomic<bool> ready(false);
    std::atomic<int> data(0);

    std::thread producer([&]() {
        data.store(42, std::memory_order_relaxed);
        ready.store(true, std::memory_order_release);  // Release: ensures data is visible
    });

    std::thread consumer([&]() {
        while (!ready.load(std::memory_order_acquire));  // Acquire: ensures we see data
        std::cout << "Consumer read data: " << data.load(std::memory_order_relaxed) << std::endl;
    });

    producer.join();
    consumer.join();
}

int main() {
    std::cout << "=== Atomic vs Non-Atomic Counter ===" << std::endl;

    const int NUM_THREADS = 10;
    const int ITERATIONS = 10000;

    std::vector<std::thread> threads;

    // Test atomic counter
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.push_back(std::thread(incrementAtomic, ITERATIONS));
    }
    for (auto& t : threads) {
        t.join();
    }

    threads.clear();

    // Test regular counter
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.push_back(std::thread(incrementRegular, ITERATIONS));
    }
    for (auto& t : threads) {
        t.join();
    }

    int expected = NUM_THREADS * ITERATIONS;
    std::cout << "\nExpected value: " << expected << std::endl;
    std::cout << "Atomic counter: " << atomicCounter << " ✓" << std::endl;
    std::cout << "Regular counter: " << regularCounter;
    if (regularCounter != expected) {
        std::cout << " ✗ (race condition!)" << std::endl;
    } else {
        std::cout << " (lucky this time)" << std::endl;
    }

    demonstrateAtomicOps();

    std::cout << "\n=== Memory Ordering ===" << std::endl;
    demonstrateMemoryOrder();

    return 0;
}
