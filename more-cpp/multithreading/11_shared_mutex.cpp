// Shared Mutex (Reader-Writer Lock)
// Multiple readers OR single writer can access data

#include <iostream>
#include <thread>
#include <shared_mutex>
#include <mutex>
#include <vector>
#include <chrono>
#include <map>

std::shared_mutex sharedMtx;
int sharedData = 0;

// Reader function (multiple readers can read simultaneously)
void reader(int id) {
    // Shared lock: multiple threads can hold this simultaneously
    std::shared_lock<std::shared_mutex> lock(sharedMtx);

    std::cout << "Reader " << id << " reading: " << sharedData << std::endl;

    // Simulate reading time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Writer function (exclusive access, blocks all readers and writers)
void writer(int id, int value) {
    // Exclusive lock: only one thread can hold this
    std::unique_lock<std::shared_mutex> lock(sharedMtx);

    std::cout << "Writer " << id << " writing: " << value << std::endl;
    sharedData = value;

    // Simulate writing time
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

// Practical example: shared cache
class ThreadSafeCache {
private:
    mutable std::shared_mutex mutex;
    std::map<std::string, int> cache;

public:
    // Read operation (shared lock)
    int get(const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(mutex);

        auto it = cache.find(key);
        if (it != cache.end()) {
            std::cout << "Cache hit: " << key << " = " << it->second << std::endl;
            return it->second;
        }

        std::cout << "Cache miss: " << key << std::endl;
        return -1;
    }

    // Write operation (exclusive lock)
    void set(const std::string& key, int value) {
        std::unique_lock<std::shared_mutex> lock(mutex);

        cache[key] = value;
        std::cout << "Cache set: " << key << " = " << value << std::endl;
    }

    // Read operation that might upgrade to write
    int getOrCompute(const std::string& key, int computedValue) {
        // First try shared lock (read)
        {
            std::shared_lock<std::shared_mutex> lock(mutex);
            auto it = cache.find(key);
            if (it != cache.end()) {
                return it->second;
            }
        }

        // Not found, need exclusive lock to write
        std::unique_lock<std::shared_mutex> lock(mutex);

        // Double-check after acquiring exclusive lock
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }

        // Insert new value
        cache[key] = computedValue;
        return computedValue;
    }
};

int main() {
    std::cout << "=== Shared Mutex (Reader-Writer Lock) ===" << std::endl;

    std::vector<std::thread> threads;

    // Start multiple readers and writers
    threads.push_back(std::thread(writer, 1, 100));

    for (int i = 1; i <= 5; i++) {
        threads.push_back(std::thread(reader, i));
    }

    threads.push_back(std::thread(writer, 2, 200));

    for (int i = 6; i <= 10; i++) {
        threads.push_back(std::thread(reader, i));
    }

    threads.push_back(std::thread(writer, 3, 300));

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "\n=== Thread-Safe Cache Example ===" << std::endl;

    ThreadSafeCache cache;

    std::thread t1([&]() { cache.set("user:1", 42); });
    std::thread t2([&]() { cache.set("user:2", 99); });
    std::thread t3([&]() { cache.get("user:1"); });
    std::thread t4([&]() { cache.get("user:2"); });
    std::thread t5([&]() { cache.get("user:3"); });

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join();

    return 0;
}
