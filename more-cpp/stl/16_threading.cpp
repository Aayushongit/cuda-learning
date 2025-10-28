/**
 * 16_threading.cpp
 *
 * THREADING AND CONCURRENCY
 * - std::thread
 * - std::mutex, std::lock_guard, std::unique_lock
 * - std::condition_variable
 * - std::future, std::promise, std::async
 * - std::atomic
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <vector>
#include <chrono>

using namespace std::chrono_literals;

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== THREADING AND CONCURRENCY ===\n";

    separator("BASIC THREADING");

    // 1. Creating Threads
    std::cout << "\n1. CREATING THREADS:\n";
    {
        auto worker = []() {
            std::cout << "Worker thread ID: " << std::this_thread::get_id() << "\n";
        };

        std::thread t1(worker);
        std::thread t2(worker);

        std::cout << "Main thread ID: " << std::this_thread::get_id() << "\n";

        t1.join();  // Wait for completion
        t2.join();
    }

    // 2. Thread with Arguments
    std::cout << "\n2. THREAD WITH ARGUMENTS:\n";
    {
        auto print_sum = [](int a, int b) {
            std::cout << "Sum: " << (a + b) << "\n";
        };

        std::thread t(print_sum, 5, 10);
        t.join();
    }

    // 3. Thread by Reference
    std::cout << "\n3. THREAD BY REFERENCE:\n";
    {
        int counter = 0;
        auto increment = [](int& c) {
            for (int i = 0; i < 5; ++i) {
                ++c;
                std::this_thread::sleep_for(10ms);
            }
        };

        std::thread t(increment, std::ref(counter));
        t.join();
        std::cout << "Counter: " << counter << "\n";
    }

    // 4. Detached Thread
    std::cout << "\n4. DETACHED THREAD:\n";
    {
        auto background_task = []() {
            std::this_thread::sleep_for(50ms);
            // Task runs in background
        };

        std::thread t(background_task);
        t.detach();  // Run independently
        // Cannot join after detach

        std::this_thread::sleep_for(100ms);  // Wait for background task
    }

    separator("MUTEXES");

    // 5. Mutex for Synchronization
    std::cout << "\n5. MUTEX:\n";
    {
        std::mutex mtx;
        int shared_counter = 0;

        auto increment_safe = [&]() {
            for (int i = 0; i < 100; ++i) {
                mtx.lock();
                ++shared_counter;
                mtx.unlock();
            }
        };

        std::thread t1(increment_safe);
        std::thread t2(increment_safe);

        t1.join();
        t2.join();

        std::cout << "Shared counter: " << shared_counter << "\n";
    }

    // 6. lock_guard (RAII)
    std::cout << "\n6. LOCK_GUARD:\n";
    {
        std::mutex mtx;
        int shared_value = 0;

        auto safe_increment = [&]() {
            for (int i = 0; i < 100; ++i) {
                std::lock_guard<std::mutex> lock(mtx);  // Auto lock/unlock
                ++shared_value;
            }  // Automatically unlocks
        };

        std::thread t1(safe_increment);
        std::thread t2(safe_increment);

        t1.join();
        t2.join();

        std::cout << "Shared value: " << shared_value << "\n";
    }

    // 7. unique_lock (flexible)
    std::cout << "\n7. UNIQUE_LOCK:\n";
    {
        std::mutex mtx;
        std::unique_lock<std::mutex> lock1(mtx);  // Locked
        // Can unlock manually
        lock1.unlock();
        // Can lock again
        lock1.lock();
        // Automatically unlocks when destroyed
        std::cout << "unique_lock provides flexible locking\n";
    }

    separator("CONDITION VARIABLES");

    // 8. Producer-Consumer
    std::cout << "\n8. PRODUCER-CONSUMER:\n";
    {
        std::mutex mtx;
        std::condition_variable cv;
        std::vector<int> buffer;
        bool done = false;

        auto producer = [&]() {
            for (int i = 1; i <= 5; ++i) {
                std::this_thread::sleep_for(50ms);
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    buffer.push_back(i);
                    std::cout << "Produced: " << i << "\n";
                }
                cv.notify_one();
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                done = true;
            }
            cv.notify_one();
        };

        auto consumer = [&]() {
            while (true) {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]() { return !buffer.empty() || done; });

                if (!buffer.empty()) {
                    int value = buffer.back();
                    buffer.pop_back();
                    lock.unlock();
                    std::cout << "  Consumed: " << value << "\n";
                } else if (done) {
                    break;
                }
            }
        };

        std::thread prod(producer);
        std::thread cons(consumer);

        prod.join();
        cons.join();
    }

    separator("ATOMICS");

    // 9. Atomic Variables
    std::cout << "\n9. ATOMIC VARIABLES:\n";
    {
        std::atomic<int> atomic_counter(0);

        auto increment_atomic = [&]() {
            for (int i = 0; i < 1000; ++i) {
                atomic_counter++;  // Thread-safe without mutex
            }
        };

        std::thread t1(increment_atomic);
        std::thread t2(increment_atomic);
        std::thread t3(increment_atomic);

        t1.join();
        t2.join();
        t3.join();

        std::cout << "Atomic counter: " << atomic_counter << "\n";
    }

    // 10. Atomic Operations
    std::cout << "\n10. ATOMIC OPERATIONS:\n";
    {
        std::atomic<int> value(10);

        std::cout << "Initial: " << value << "\n";
        value.store(20);
        std::cout << "After store(20): " << value.load() << "\n";

        int old = value.exchange(30);
        std::cout << "exchange(30) returned: " << old << ", new value: " << value << "\n";

        int expected = 30;
        bool success = value.compare_exchange_strong(expected, 40);
        std::cout << "compare_exchange_strong: " << (success ? "success" : "failed") << "\n";
        std::cout << "Value: " << value << "\n";
    }

    separator("ASYNC AND FUTURES");

    // 11. std::async
    std::cout << "\n11. STD::ASYNC:\n";
    {
        auto compute = [](int x, int y) {
            std::this_thread::sleep_for(100ms);
            return x + y;
        };

        std::future<int> result = std::async(std::launch::async, compute, 5, 10);
        std::cout << "Computing in background...\n";
        std::cout << "Result: " << result.get() << "\n";  // Blocks until ready
    }

    // 12. Multiple Async Tasks
    std::cout << "\n12. MULTIPLE ASYNC TASKS:\n";
    {
        auto task = [](int id) {
            std::this_thread::sleep_for(50ms);
            return id * id;
        };

        std::vector<std::future<int>> futures;
        for (int i = 1; i <= 5; ++i) {
            futures.push_back(std::async(std::launch::async, task, i));
        }

        std::cout << "Results: ";
        for (auto& f : futures) {
            std::cout << f.get() << " ";
        }
        std::cout << "\n";
    }

    // 13. Promise and Future
    std::cout << "\n13. PROMISE AND FUTURE:\n";
    {
        std::promise<int> prom;
        std::future<int> fut = prom.get_future();

        std::thread t([&prom]() {
            std::this_thread::sleep_for(100ms);
            prom.set_value(42);  // Set result
        });

        std::cout << "Waiting for result...\n";
        std::cout << "Got: " << fut.get() << "\n";

        t.join();
    }

    // 14. Exception Handling with Future
    std::cout << "\n14. EXCEPTION IN ASYNC:\n";
    {
        auto throwing_task = []() {
            std::this_thread::sleep_for(50ms);
            throw std::runtime_error("Error in async task");
            return 42;
        };

        std::future<int> result = std::async(std::launch::async, throwing_task);

        try {
            result.get();
        } catch (const std::exception& e) {
            std::cout << "Caught: " << e.what() << "\n";
        }
    }

    separator("THREAD POOL PATTERN");

    // 15. Simple Thread Pool
    std::cout << "\n15. SIMPLE THREAD POOL:\n";
    {
        std::vector<std::thread> pool;
        std::mutex mtx;

        auto worker = [&](int id) {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "Worker " << id << " processing\n";
        };

        for (int i = 0; i < 4; ++i) {
            pool.emplace_back(worker, i);
        }

        for (auto& t : pool) {
            t.join();
        }
    }

    separator("THREAD SAFETY TIPS");

    std::cout << "\n1. Always join or detach threads\n";
    std::cout << "2. Use RAII wrappers (lock_guard, unique_lock)\n";
    std::cout << "3. Prefer atomic operations over mutexes for simple counters\n";
    std::cout << "4. Use condition variables for event notification\n";
    std::cout << "5. Avoid deadlocks: acquire locks in consistent order\n";
    std::cout << "6. Use std::async for simple parallel tasks\n";
    std::cout << "7. Be careful with data races\n";

    std::cout << "\n=== END OF THREADING ===\n";

    return 0;
}
