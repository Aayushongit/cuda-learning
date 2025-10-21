#include <iostream>
#include <thread>
#include <vector>
#include <string>

struct ThreadData {
    int thread_id;
    std::string message;
};

void worker(const ThreadData& data) {
    std::cout << "Thread " << data.thread_id << ": " << data.message << "\n";
}

int main() {
    const int NUM_THREADS = 5;
    std::vector<std::thread> threads;
    std::vector<ThreadData> thread_data;

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data.push_back({i, "Processing task " + std::to_string(i)});
    }

    for (const auto& data : thread_data) {
        threads.emplace_back(worker, std::cref(data));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "All threads completed\n";
    return 0;
}
