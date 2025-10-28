// Thread IDs and Hardware Information
// Shows how to get thread IDs and hardware concurrency

#include <iostream>
#include <thread>
#include <vector>

void printThreadInfo(int threadNum) {
    // Get the ID of the current thread
    std::thread::id this_id = std::this_thread::get_id();
    std::cout << "Thread " << threadNum << " ID: " << this_id << std::endl;
}

int main() {
    std::cout << "=== Hardware Concurrency ===" << std::endl;

    // Get number of concurrent threads supported by hardware
    unsigned int numThreads = std::thread::hardware_concurrency();
    std::cout << "Hardware supports " << numThreads << " concurrent threads" << std::endl;

    std::cout << "\n=== Thread IDs ===" << std::endl;

    // Main thread ID
    std::cout << "Main thread ID: " << std::this_thread::get_id() << std::endl;

    // Create multiple threads
    std::vector<std::thread> threads;
    for (int i = 0; i < 5; i++) {
        threads.push_back(std::thread(printThreadInfo, i + 1));
    }

    // Join all threads
    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
