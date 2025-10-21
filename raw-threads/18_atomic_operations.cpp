#include <iostream>
#include <thread>
#include <atomic>
#include <vector>

const int NUM_THREADS = 10;
const int INCREMENTS = 100000;

std::atomic<int> atomic_counter(0);
int normal_counter = 0;

void atomic_increment() {
    for (int i = 0; i < INCREMENTS; i++) {
        atomic_counter.fetch_add(1, std::memory_order_relaxed);
        normal_counter++;
    }
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(atomic_increment);
    }

    for (auto& t : threads) {
        t.join();
    }

    int expected = NUM_THREADS * INCREMENTS;
    std::cout << "Expected:        " << expected << "\n";
    std::cout << "Atomic counter:  " << atomic_counter.load() << " (correct)\n";
    std::cout << "Normal counter:  " << normal_counter << " (race condition)\n";

    return 0;
}
