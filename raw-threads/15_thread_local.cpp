#include <iostream>
#include <thread>
#include <vector>

thread_local int local_data = 0;

void worker(int id) {
    local_data = id * 100;

    std::cout << "Thread " << id << ": Set thread-local value to " << local_data << "\n";

    std::cout << "Thread " << id << ": Retrieved thread-local value: " << local_data << "\n";
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 1; i <= 3; i++) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Main thread local_data: " << local_data << "\n";

    return 0;
}
