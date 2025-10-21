#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

std::mutex mtx;
int shared_resource = 0;

void safe_increment() {
    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lock(mtx);
        shared_resource++;
    }
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; i++) {
        threads.emplace_back(safe_increment);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Final value: " << shared_resource << " (expected: 1000000)\n";

    return 0;
}
