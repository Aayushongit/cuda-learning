#include <iostream>
#include <thread>
#include <shared_mutex>
#include <vector>
#include <chrono>

std::shared_mutex rw_mutex;
int shared_data = 0;

void reader(int id) {
    for (int i = 0; i < 3; i++) {
        std::shared_lock<std::shared_mutex> lock(rw_mutex);

        std::cout << "Reader " << id << ": reading value " << shared_data << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void writer(int id) {
    for (int i = 0; i < 3; i++) {
        std::unique_lock<std::shared_mutex> lock(rw_mutex);

        shared_data++;
        std::cout << "Writer " << id << ": wrote value " << shared_data << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
    }
}

int main() {
    std::vector<std::thread> threads;

    for (int i = 1; i <= 3; i++) {
        threads.emplace_back(reader, i);
    }

    for (int i = 1; i <= 2; i++) {
        threads.emplace_back(writer, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
