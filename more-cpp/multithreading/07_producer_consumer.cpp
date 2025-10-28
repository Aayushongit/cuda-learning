// Producer-Consumer Pattern
// Classic multithreading pattern using queue, mutex, and condition variables

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <vector>

// Shared queue between producers and consumers
std::queue<int> dataQueue;
std::mutex queueMutex;
std::condition_variable queueCV;

const int MAX_QUEUE_SIZE = 5;
bool finished = false;

// Producer: generates data and puts it in the queue
void producer(int id, int itemCount) {
    for (int i = 1; i <= itemCount; i++) {
        std::unique_lock<std::mutex> lock(queueMutex);

        // Wait if queue is full
        queueCV.wait(lock, []{ return dataQueue.size() < MAX_QUEUE_SIZE; });

        int item = id * 100 + i;
        dataQueue.push(item);
        std::cout << "Producer " << id << " produced: " << item
                  << " (Queue size: " << dataQueue.size() << ")" << std::endl;

        lock.unlock();

        // Notify consumers that data is available
        queueCV.notify_all();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Consumer: takes data from the queue and processes it
void consumer(int id) {
    while (true) {
        std::unique_lock<std::mutex> lock(queueMutex);

        // Wait until queue has data or production is finished
        queueCV.wait(lock, []{ return !dataQueue.empty() || finished; });

        // Exit if finished and queue is empty
        if (finished && dataQueue.empty()) {
            std::cout << "Consumer " << id << " exiting" << std::endl;
            break;
        }

        if (!dataQueue.empty()) {
            int item = dataQueue.front();
            dataQueue.pop();
            std::cout << "Consumer " << id << " consumed: " << item
                      << " (Queue size: " << dataQueue.size() << ")" << std::endl;

            lock.unlock();

            // Notify producers that space is available
            queueCV.notify_all();

            // Simulate processing time
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
    }
}

int main() {
    std::cout << "=== Producer-Consumer Pattern ===" << std::endl;
    std::cout << "Max queue size: " << MAX_QUEUE_SIZE << "\n" << std::endl;

    // Create multiple producers and consumers
    std::vector<std::thread> threads;

    // 2 producers, each producing 10 items
    threads.push_back(std::thread(producer, 1, 10));
    threads.push_back(std::thread(producer, 2, 10));

    // 3 consumers
    threads.push_back(std::thread(consumer, 1));
    threads.push_back(std::thread(consumer, 2));
    threads.push_back(std::thread(consumer, 3));

    // Wait for producers to finish
    threads[0].join();
    threads[1].join();

    // Signal that production is finished
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        finished = true;
    }
    queueCV.notify_all();

    // Wait for consumers to finish
    for (size_t i = 2; i < threads.size(); i++) {
        threads[i].join();
    }

    std::cout << "\nAll production and consumption completed!" << std::endl;
    return 0;
}
