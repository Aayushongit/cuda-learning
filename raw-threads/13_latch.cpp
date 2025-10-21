#include <iostream>
#include <thread>
#include <vector>
#include <latch>
#include <chrono>

const int NUM_THREADS = 4;

void worker(int id, std::latch& sync_point) {
    std::cout << "Thread " << id << ": Phase 1 - Initialization\n";
    std::this_thread::sleep_for(std::chrono::seconds(id));

    std::cout << "Thread " << id << ": Waiting at latch\n";
    sync_point.arrive_and_wait();

    std::cout << "Thread " << id << ": Phase 2 - All threads synchronized\n";
}

int main() {
    std::latch sync_point(NUM_THREADS);
    std::vector<std::thread> threads;

    for (int i = 1; i <= NUM_THREADS; i++) {
        threads.emplace_back(worker, i, std::ref(sync_point));
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
