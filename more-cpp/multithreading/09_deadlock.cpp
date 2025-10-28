// Deadlock Demonstration and Prevention
// Shows how deadlock occurs and how to prevent it

#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

std::mutex mutex1;
std::mutex mutex2;

// DEADLOCK EXAMPLE: Each thread waits for the other's lock
void deadlockThread1() {
    std::lock_guard<std::mutex> lock1(mutex1);
    std::cout << "Thread 1: Locked mutex1" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::cout << "Thread 1: Trying to lock mutex2..." << std::endl;
    std::lock_guard<std::mutex> lock2(mutex2);  // Will wait forever!
    std::cout << "Thread 1: Locked both mutexes" << std::endl;
}

void deadlockThread2() {
    std::lock_guard<std::mutex> lock2(mutex2);
    std::cout << "Thread 2: Locked mutex2" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::cout << "Thread 2: Trying to lock mutex1..." << std::endl;
    std::lock_guard<std::mutex> lock1(mutex1);  // Will wait forever!
    std::cout << "Thread 2: Locked both mutexes" << std::endl;
}

// SOLUTION 1: Lock in the same order
void safeThread1() {
    std::lock_guard<std::mutex> lock1(mutex1);
    std::cout << "Safe Thread 1: Locked mutex1" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::lock_guard<std::mutex> lock2(mutex2);
    std::cout << "Safe Thread 1: Locked both mutexes" << std::endl;
}

void safeThread2() {
    // Same order as thread1: mutex1 then mutex2
    std::lock_guard<std::mutex> lock1(mutex1);
    std::cout << "Safe Thread 2: Locked mutex1" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::lock_guard<std::mutex> lock2(mutex2);
    std::cout << "Safe Thread 2: Locked both mutexes" << std::endl;
}

// SOLUTION 2: Use std::lock to lock multiple mutexes atomically
void safeLockThread1() {
    std::unique_lock<std::mutex> lock1(mutex1, std::defer_lock);
    std::unique_lock<std::mutex> lock2(mutex2, std::defer_lock);

    // Lock both atomically (no deadlock possible)
    std::lock(lock1, lock2);

    std::cout << "Std::lock Thread 1: Locked both mutexes safely" << std::endl;
}

void safeLockThread2() {
    std::unique_lock<std::mutex> lock1(mutex1, std::defer_lock);
    std::unique_lock<std::mutex> lock2(mutex2, std::defer_lock);

    std::lock(lock1, lock2);

    std::cout << "Std::lock Thread 2: Locked both mutexes safely" << std::endl;
}

// SOLUTION 3: Use scoped_lock (C++17) - easiest and best
void scopedLockThread1() {
    std::scoped_lock lock(mutex1, mutex2);
    std::cout << "Scoped_lock Thread 1: Locked both mutexes" << std::endl;
}

void scopedLockThread2() {
    std::scoped_lock lock(mutex1, mutex2);
    std::cout << "Scoped_lock Thread 2: Locked both mutexes" << std::endl;
}

int main() {
    std::cout << "=== Deadlock Prevention ===" << std::endl;

    // WARNING: Uncommenting this will cause deadlock!
    // std::cout << "\n--- DEADLOCK EXAMPLE (COMMENTED OUT) ---" << std::endl;
    // std::thread t1(deadlockThread1);
    // std::thread t2(deadlockThread2);
    // t1.join();
    // t2.join();

    std::cout << "\n--- Solution 1: Same Lock Order ---" << std::endl;
    std::thread t3(safeThread1);
    std::thread t4(safeThread2);
    t3.join();
    t4.join();

    std::cout << "\n--- Solution 2: std::lock ---" << std::endl;
    std::thread t5(safeLockThread1);
    std::thread t6(safeLockThread2);
    t5.join();
    t6.join();

    std::cout << "\n--- Solution 3: scoped_lock (C++17) ---" << std::endl;
    std::thread t7(scopedLockThread1);
    std::thread t8(scopedLockThread2);
    t7.join();
    t8.join();

    std::cout << "\nAll threads completed successfully!" << std::endl;
    return 0;
}
