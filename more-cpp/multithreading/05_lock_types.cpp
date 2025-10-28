// Different Lock Types in C++
// Demonstrates lock_guard, unique_lock, and scoped_lock

#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

std::mutex mtx1, mtx2;
int sharedData = 0;

// 1. lock_guard: Simple RAII lock (cannot be unlocked manually)
void useLockGuard() {
    std::lock_guard<std::mutex> lock(mtx1);
    sharedData++;
    // Automatically unlocks when function exits
}

// 2. unique_lock: More flexible, can be unlocked/locked manually
void useUniqueLock() {
    std::unique_lock<std::mutex> lock(mtx1);
    sharedData++;

    // Can unlock manually if needed
    lock.unlock();

    // Do some work without holding the lock
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Can relock
    lock.lock();
    sharedData++;
    // Automatically unlocks on destruction
}

// 3. scoped_lock: Lock multiple mutexes safely (C++17)
void useScopedLock() {
    // Locks both mutexes atomically, prevents deadlock
    std::scoped_lock lock(mtx1, mtx2);
    sharedData++;
    // Both mutexes unlock automatically
}

// Deferred locking with unique_lock
void useDeferredLock() {
    // Create lock without locking immediately
    std::unique_lock<std::mutex> lock(mtx1, std::defer_lock);

    // Do some work before locking
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Now lock when needed
    lock.lock();
    sharedData++;
}

// Try to lock without blocking
void useTryLock() {
    std::unique_lock<std::mutex> lock(mtx1, std::defer_lock);

    if (lock.try_lock()) {
        std::cout << "Lock acquired successfully" << std::endl;
        sharedData++;
    } else {
        std::cout << "Failed to acquire lock" << std::endl;
    }
}

int main() {
    std::cout << "=== Lock Types Demonstration ===" << std::endl;

    std::cout << "\n1. lock_guard (simple RAII):" << std::endl;
    useLockGuard();
    std::cout << "   Data after lock_guard: " << sharedData << std::endl;

    std::cout << "\n2. unique_lock (flexible):" << std::endl;
    useUniqueLock();
    std::cout << "   Data after unique_lock: " << sharedData << std::endl;

    std::cout << "\n3. scoped_lock (multiple mutexes):" << std::endl;
    useScopedLock();
    std::cout << "   Data after scoped_lock: " << sharedData << std::endl;

    std::cout << "\n4. Deferred locking:" << std::endl;
    useDeferredLock();
    std::cout << "   Data after deferred lock: " << sharedData << std::endl;

    std::cout << "\n5. Try lock:" << std::endl;
    useTryLock();

    return 0;
}
