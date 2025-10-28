// Packaged Task
// Wraps a callable object to be executed asynchronously

#include <iostream>
#include <thread>
#include <future>
#include <queue>
#include <functional>

// Simple function to package
int add(int a, int b) {
    std::cout << "Computing " << a << " + " << b << std::endl;
    return a + b;
}

// Task queue system using packaged_task
class TaskQueue {
private:
    std::queue<std::packaged_task<int()>> tasks;
    std::mutex mtx;
    bool running = true;

public:
    // Add task to queue
    std::future<int> addTask(std::packaged_task<int()>&& task) {
        std::future<int> result = task.get_future();

        std::lock_guard<std::mutex> lock(mtx);
        tasks.push(std::move(task));

        return result;
    }

    // Worker that processes tasks
    void worker(int id) {
        while (running) {
            std::packaged_task<int()> task;

            {
                std::lock_guard<std::mutex> lock(mtx);
                if (tasks.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                task = std::move(tasks.front());
                tasks.pop();
            }

            std::cout << "Worker " << id << " executing task" << std::endl;
            task();  // Execute the task
        }
    }

    void stop() {
        running = false;
    }
};

int main() {
    std::cout << "=== Packaged Task Basics ===" << std::endl;

    // Create a packaged task
    std::packaged_task<int(int, int)> task1(add);

    // Get future before moving the task
    std::future<int> result1 = task1.get_future();

    // Execute in another thread
    std::thread t1(std::move(task1), 10, 20);

    // Get result
    std::cout << "Result: " << result1.get() << std::endl;
    t1.join();

    std::cout << "\n=== Packaged Task with Lambda ===" << std::endl;

    // Package a lambda
    std::packaged_task<int()> task2([]() {
        std::cout << "Lambda task executing..." << std::endl;
        return 42;
    });

    std::future<int> result2 = task2.get_future();

    // Execute in current thread
    task2();

    std::cout << "Lambda result: " << result2.get() << std::endl;

    std::cout << "\n=== Task Queue Example ===" << std::endl;

    TaskQueue queue;

    // Start worker threads
    std::thread worker1(&TaskQueue::worker, &queue, 1);
    std::thread worker2(&TaskQueue::worker, &queue, 2);

    // Submit tasks
    std::vector<std::future<int>> futures;

    for (int i = 1; i <= 5; i++) {
        std::packaged_task<int()> task([i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            return i * i;
        });

        futures.push_back(queue.addTask(std::move(task)));
    }

    // Collect results
    std::cout << "\nCollecting results:" << std::endl;
    for (size_t i = 0; i < futures.size(); i++) {
        std::cout << "Task " << (i + 1) << " result: " << futures[i].get() << std::endl;
    }

    // Stop workers
    queue.stop();
    worker1.join();
    worker2.join();

    std::cout << "\n=== Exception Handling ===" << std::endl;

    std::packaged_task<int()> task3([]() -> int {
        throw std::runtime_error("Task failed!");
        return 0;
    });

    std::future<int> result3 = task3.get_future();
    task3();

    try {
        result3.get();
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }

    std::cout << "\n=== Resetting Packaged Task ===" << std::endl;

    // Packaged task can be reset and reused
    std::packaged_task<int(int)> reusableTask([](int x) {
        return x * 2;
    });

    // First use
    auto fut1 = reusableTask.get_future();
    reusableTask(5);
    std::cout << "First result: " << fut1.get() << std::endl;

    // Reset and reuse
    reusableTask.reset();
    auto fut2 = reusableTask.get_future();
    reusableTask(10);
    std::cout << "Second result: " << fut2.get() << std::endl;

    return 0;
}
