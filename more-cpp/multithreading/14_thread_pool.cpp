// Thread Pool Implementation
// Reuses threads to execute multiple tasks efficiently

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <vector>
#include <future>

class ThreadPool {
private:
    std::vector<std::thread> workers;           // Worker threads
    std::queue<std::function<void()>> tasks;    // Task queue
    std::mutex queueMutex;                      // Protects task queue
    std::condition_variable condition;          // Notifies workers
    bool stop;                                  // Stop flag

public:
    // Constructor: create thread pool with given size
    ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; i++) {
            workers.emplace_back([this, i] {
                std::cout << "Worker " << i << " started" << std::endl;

                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queueMutex);

                        // Wait for task or stop signal
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });

                        // Exit if stopped and no tasks left
                        if (stop && tasks.empty()) {
                            std::cout << "Worker " << i << " exiting" << std::endl;
                            return;
                        }

                        // Get next task
                        task = std::move(tasks.front());
                        tasks.pop();
                    }

                    // Execute task
                    task();
                }
            });
        }
    }

    // Submit a task to the pool
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        // Create packaged task
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> result = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queueMutex);

            if (stop) {
                throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
            }

            tasks.emplace([task]() { (*task)(); });
        }

        condition.notify_one();
        return result;
    }

    // Destructor: stop all threads
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }

        condition.notify_all();

        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    // Get number of pending tasks
    size_t pendingTasks() {
        std::unique_lock<std::mutex> lock(queueMutex);
        return tasks.size();
    }
};

// Example tasks
int multiply(int a, int b) {
    std::cout << "Calculating " << a << " * " << b << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return a * b;
}

void printMessage(const std::string& msg) {
    std::cout << "Message: " << msg << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

int main() {
    std::cout << "=== Thread Pool Example ===" << std::endl;

    // Create pool with 4 worker threads
    ThreadPool pool(4);

    std::cout << "\nSubmitting 10 tasks to pool of 4 threads...\n" << std::endl;

    // Submit tasks and collect futures
    std::vector<std::future<int>> results;

    for (int i = 1; i <= 10; i++) {
        results.push_back(pool.enqueue(multiply, i, i));
    }

    // Submit tasks without return value
    pool.enqueue(printMessage, "Hello from thread pool!");
    pool.enqueue(printMessage, "Tasks are executed by worker threads");

    std::cout << "Pending tasks: " << pool.pendingTasks() << std::endl;

    // Get results
    std::cout << "\nCollecting results:" << std::endl;
    for (size_t i = 0; i < results.size(); i++) {
        std::cout << "Result " << (i + 1) << ": " << results[i].get() << std::endl;
    }

    std::cout << "\nAll tasks completed!" << std::endl;

    // Pool destructor will be called here, cleaning up threads

    return 0;
}
