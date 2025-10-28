// Race Condition Demonstration
// Shows the problem when multiple threads access shared data without synchronization

#include <iostream>
#include <thread>
#include <vector>

// Shared counter (UNSAFE without synchronization)
int unsafeCounter = 0;

// Function that increments counter (NOT thread-safe)
void incrementUnsafe(int iterations) {
    for (int i = 0; i < iterations; i++) {
        // This is NOT atomic: read, increment, write
        unsafeCounter++;
    }
}

int main() {
    std::cout << "=== Race Condition Demo ===" << std::endl;
    std::cout << "Multiple threads incrementing shared counter without protection\n" << std::endl;

    const int NUM_THREADS = 10;
    const int ITERATIONS = 10000;

    std::vector<std::thread> threads;

    // Create threads that all modify the same counter
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.push_back(std::thread(incrementUnsafe, ITERATIONS));
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    // Expected value: NUM_THREADS * ITERATIONS
    int expected = NUM_THREADS * ITERATIONS;
    std::cout << "Expected counter value: " << expected << std::endl;
    std::cout << "Actual counter value: " << unsafeCounter << std::endl;

    if (unsafeCounter != expected) {
        std::cout << "\nRACE CONDITION DETECTED!" << std::endl;
        std::cout << "Lost updates: " << (expected - unsafeCounter) << std::endl;
        std::cout << "This happened because multiple threads modified the counter simultaneously" << std::endl;
    } else {
        std::cout << "\nNo race condition this time (try running again)" << std::endl;
    }

    return 0;
}
