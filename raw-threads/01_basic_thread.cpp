#include <iostream>
#include <thread>
#include <chrono>

void thread_function() {
    std::cout << "Hello from thread!\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Thread finishing...\n";
}

int main() {
    std::cout << "Main: Creating thread\n";
    std::thread t(thread_function);

    std::cout << "Main: Waiting for thread to finish\n";
    t.join();

    std::cout << "Main: Thread completed\n";
    return 0;
}
