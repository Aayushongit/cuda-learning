#include <iostream>
#include <thread>
#include <chrono>

void detached_worker(int id) {
    std::cout << "Detached thread " << id << " starting\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Detached thread " << id << " finishing\n";
}

void joined_worker(int id) {
    std::cout << "Joined thread " << id << " starting\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Joined thread " << id << " finishing\n";
}

int main() {
    std::thread detached_thread(detached_worker, 1);
    detached_thread.detach();

    std::thread joined_thread(joined_worker, 2);
    joined_thread.join();
    std::cout << "Main: Joined thread completed\n";

    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << "Main: Exiting (detached thread runs independently)\n";

    return 0;
}
