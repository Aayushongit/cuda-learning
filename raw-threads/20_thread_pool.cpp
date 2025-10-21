#include <iostream>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; i++) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });

                        if (stop && tasks.empty()) {
                            return;
                        }

                        task = std::move(tasks.front());
                        tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    template<typename F>
    void enqueue(F&& f) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();

        for (auto& worker : workers) {
            worker.join();
        }
    }
};

void execute_task(int arg) {
    std::cout << "Thread " << std::this_thread::get_id()
              << ": Processing task " << arg << "\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

int main() {
    ThreadPool pool(4);

    for (int i = 1; i <= 8; i++) {
        pool.enqueue([i] { execute_task(i); });
        std::cout << "Main: Added task " << i << " to queue\n";
    }

    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "All tasks completed, thread pool shutting down\n";

    return 0;
}
