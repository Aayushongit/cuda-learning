// Condition Variables - Thread Synchronization
// Allows threads to wait for certain conditions to be met

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

std::mutex mtx;
std::condition_variable cv;
bool dataReady = false;
int sharedData = 0;

// Thread that waits for data to be ready
void consumer() {
    std::unique_lock<std::mutex> lock(mtx);

    std::cout << "Consumer: Waiting for data..." << std::endl;

    // Wait until dataReady becomes true
    // This releases the lock and waits, then reacquires when notified
    cv.wait(lock, []{ return dataReady; });

    std::cout << "Consumer: Data received! Value = " << sharedData << std::endl;
}

// Thread that produces data
void producer() {
    std::this_thread::sleep_for(std::chrono::seconds(2));

    {
        std::lock_guard<std::mutex> lock(mtx);
        sharedData = 42;
        dataReady = true;
        std::cout << "Producer: Data is ready!" << std::endl;
    }

    // Notify the waiting thread
    cv.notify_one();
}

// Example with wait_for (timeout)
void consumerWithTimeout() {
    std::unique_lock<std::mutex> lock(mtx);

    std::cout << "Consumer: Waiting with timeout..." << std::endl;

    // Wait for maximum 1 second
    if (cv.wait_for(lock, std::chrono::seconds(1), []{ return dataReady; })) {
        std::cout << "Consumer: Got data within timeout!" << std::endl;
    } else {
        std::cout << "Consumer: Timeout! No data received." << std::endl;
    }
}

int main() {
    std::cout << "=== Condition Variable Demo ===" << std::endl;

    std::cout << "\n--- Example 1: Basic wait/notify ---" << std::endl;
    std::thread t1(consumer);
    std::thread t2(producer);

    t1.join();
    t2.join();

    // Reset for second example
    dataReady = false;

    std::cout << "\n--- Example 2: Wait with timeout ---" << std::endl;
    std::thread t3(consumerWithTimeout);
    // Note: No producer this time, so it will timeout
    t3.join();

    return 0;
}
