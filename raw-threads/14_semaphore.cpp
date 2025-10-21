#include <iostream>
#include <thread>
#include <vector>
#include <semaphore>
#include <chrono>

const int MAX_RESOURCES = 3;
const int NUM_THREADS = 5;

std::counting_semaphore<MAX_RESOURCES> semaphore(MAX_RESOURCES);

void use_resource(int id) {
    std::cout << "Thread " << id << ": Waiting for resource\n";
    semaphore.acquire();

    std::cout << "Thread " << id << ": Acquired resource\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Thread " << id << ": Releasing resource\n";

    semaphore.release();
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 1; i <= NUM_THREADS; i++) {
        threads.emplace_back(use_resource, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
