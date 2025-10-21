#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void waiter() {
    std::unique_lock<std::mutex> lock(mtx);

    std::cout << "Waiter: Waiting for signal...\n";
    cv.wait(lock, []{ return ready; });
    std::cout << "Waiter: Received signal! Proceeding...\n";
}

void signaler() {
    std::this_thread::sleep_for(std::chrono::seconds(2));

    {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Signaler: Setting ready flag and signaling\n";
        ready = true;
    }
    cv.notify_one();
}

int main() {
    std::thread t1(waiter);
    std::thread t2(signaler);

    t1.join();
    t2.join();

    return 0;
}
