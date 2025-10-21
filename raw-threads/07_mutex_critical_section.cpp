#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

std::mutex mtx;

void print_pattern(const std::string& pattern) {
    std::lock_guard<std::mutex> lock(mtx);

    for (int i = 0; i < 5; i++) {
        std::cout << pattern;
        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "\n";
}

int main() {
    std::thread t1(print_pattern, "A");
    std::thread t2(print_pattern, "B");
    std::thread t3(print_pattern, "C");

    t1.join();
    t2.join();
    t3.join();

    return 0;
}
