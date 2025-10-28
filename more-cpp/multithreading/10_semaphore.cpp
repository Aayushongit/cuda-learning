// Semaphore Example (C++20)
// Semaphore controls access to a limited resource pool

#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <semaphore>

// Semaphore with 3 permits (allows 3 concurrent accesses)
std::counting_semaphore<3> semaphore(3);

// Binary semaphore (like a mutex, 0 or 1)
std::binary_semaphore binarySem(1);

// Simulate accessing a limited resource (e.g., database connections)
void accessLimitedResource(int id) {
    std::cout << "Thread " << id << ": Waiting for resource..." << std::endl;

    // Acquire semaphore (decrements counter, blocks if 0)
    semaphore.acquire();

    std::cout << "Thread " << id << ": Got resource, working..." << std::endl;

    // Simulate work
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "Thread " << id << ": Releasing resource" << std::endl;

    // Release semaphore (increments counter)
    semaphore.release();
}

// Binary semaphore example (similar to mutex)
int sharedCounter = 0;

void incrementWithBinarySem(int iterations) {
    for (int i = 0; i < iterations; i++) {
        binarySem.acquire();
        sharedCounter++;
        binarySem.release();
    }
}

int main() {
    std::cout << "=== Counting Semaphore (max 3 concurrent) ===" << std::endl;

    // Create 6 threads, but only 3 can access resource at once
    std::vector<std::thread> threads;
    for (int i = 1; i <= 6; i++) {
        threads.push_back(std::thread(accessLimitedResource, i));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "\n=== Binary Semaphore ===" << std::endl;

    threads.clear();
    for (int i = 0; i < 5; i++) {
        threads.push_back(std::thread(incrementWithBinarySem, 1000));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Counter value: " << sharedCounter << std::endl;
    std::cout << "Expected: " << 5 * 1000 << std::endl;

    return 0;
}
