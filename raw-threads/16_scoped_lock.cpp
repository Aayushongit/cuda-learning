#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

std::mutex mutex1;
std::mutex mutex2;

void safe_thread1() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "Thread 1: Trying to lock both mutexes\n";
    std::scoped_lock lock(mutex1, mutex2);

    std::cout << "Thread 1: Got both locks! (no deadlock)\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

void safe_thread2() {
    std::cout << "Thread 2: Trying to lock both mutexes\n";
    std::scoped_lock lock(mutex2, mutex1);

    std::cout << "Thread 2: Got both locks! (no deadlock)\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

int main() {
    std::cout << "Using std::scoped_lock to prevent deadlock\n";
    std::thread t1(safe_thread1);
    std::thread t2(safe_thread2);

    t1.join();
    t2.join();

    std::cout << "Both threads completed successfully!\n";

    return 0;
}
