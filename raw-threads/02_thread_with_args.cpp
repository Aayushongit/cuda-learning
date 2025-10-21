#include <iostream>
#include <thread>
#include <vector>

void print_number(int num) {
    std::cout << "Thread received: " << num << "\n";
    std::cout << "Thread ID: " << std::this_thread::get_id() << "\n";
}

int main() {
    std::vector<std::thread> threads;
    int numbers[] = {10, 20, 30};

    for (int num : numbers) {
        threads.emplace_back(print_number, num);
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
