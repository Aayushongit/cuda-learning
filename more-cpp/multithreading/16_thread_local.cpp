// Thread Local Storage
// Each thread has its own copy of thread_local variables

#include <iostream>
#include <thread>
#include <vector>
#include <sstream>
#include <random>
#include <chrono>

// Regular global variable (shared by all threads)
int globalCounter = 0;

// Thread-local variable (each thread has its own copy)
thread_local int threadLocalCounter = 0;

// Thread-local with initialization
thread_local int threadId = -1;

void incrementCounters(int id) {
    // Set thread-local ID on first access
    if (threadId == -1) {
        threadId = id;
    }

    for (int i = 0; i < 5; i++) {
        globalCounter++;           // Shared (race condition!)
        threadLocalCounter++;      // Thread-local (safe)

        std::ostringstream oss;
        oss << "Thread " << threadId
            << ": global=" << globalCounter
            << ", thread_local=" << threadLocalCounter << std::endl;
        std::cout << oss.str();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Thread " << threadId << " final thread_local value: "
              << threadLocalCounter << std::endl;
}

// Thread-local object with constructor/destructor
class ThreadResource {
public:
    ThreadResource(int id) : id(id) {
        std::cout << "ThreadResource " << id << " constructed in thread "
                  << std::this_thread::get_id() << std::endl;
    }

    ~ThreadResource() {
        std::cout << "ThreadResource " << id << " destroyed in thread "
                  << std::this_thread::get_id() << std::endl;
    }

    void use() {
        std::cout << "Using ThreadResource " << id << std::endl;
    }

private:
    int id;
};

thread_local ThreadResource resource(42);

void useThreadResource(int id) {
    std::cout << "Thread " << id << " using resource" << std::endl;
    resource.use();
}

// Practical example: thread-local random number generator
thread_local std::mt19937 generator(std::random_device{}());

int getRandomNumber(int min, int max) {
    std::uniform_int_distribution<> dist(min, max);
    return dist(generator);
}

void generateRandomNumbers(int id) {
    for (int i = 0; i < 3; i++) {
        int num = getRandomNumber(1, 100);
        std::cout << "Thread " << id << " generated: " << num << std::endl;
    }
}

int main() {
    std::cout << "=== Thread Local Storage ===" << std::endl;

    std::cout << "\n--- Counter Example ---" << std::endl;
    std::vector<std::thread> threads;

    for (int i = 1; i <= 3; i++) {
        threads.push_back(std::thread(incrementCounters, i));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "\nFinal global counter (with race condition): " << globalCounter << std::endl;
    std::cout << "Main thread's thread_local counter: " << threadLocalCounter << std::endl;

    std::cout << "\n--- Thread Resource Example ---" << std::endl;
    threads.clear();

    for (int i = 1; i <= 3; i++) {
        threads.push_back(std::thread(useThreadResource, i));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "\n--- Thread-Local Random Generator ---" << std::endl;
    threads.clear();

    for (int i = 1; i <= 3; i++) {
        threads.push_back(std::thread(generateRandomNumbers, i));
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
