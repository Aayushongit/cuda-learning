#include <iostream>
#include <thread>
#include <vector>

const int NUM_THREADS = 10;
const int INCREMENTS = 100000;

int counter = 0;

void increment_counter() {
    for (int i = 0; i < INCREMENTS; i++) {
        counter++;  // NOT thread-safe!
    }
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(increment_counter);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Expected: " << NUM_THREADS * INCREMENTS << "\n";
    std::cout << "Actual:   " << counter << "\n";
    std::cout << "Lost:     " << (NUM_THREADS * INCREMENTS) - counter << "\n";

    return 0;
}
