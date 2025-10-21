#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

std::mutex mutex1;
std::mutex mutex2;

void thread1_func() {
    std::cout << "Thread 1: Locking mutex1\n";
    std::lock_guard<std::mutex> lock1(mutex1);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "Thread 1: Trying to lock mutex2\n";
    std::lock_guard<std::mutex> lock2(mutex2);

    std::cout << "Thread 1: Got both locks!\n";
}

void thread2_func() {
    std::cout << "Thread 2: Locking mutex2\n";
    std::lock_guard<std::mutex> lock2(mutex2);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "Thread 2: Trying to lock mutex1\n";
    std::lock_guard<std::mutex> lock1(mutex1);

    std::cout << "Thread 2: Got both locks!\n";
}

int main() {
    std::cout << "Starting threads (will deadlock)...\n";
    std::thread t1(thread1_func);
    std::thread t2(thread2_func);

    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "Deadlock occurred - program stuck!\n";

    return 0;
}
