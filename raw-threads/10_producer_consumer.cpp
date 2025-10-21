#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

const int BUFFER_SIZE = 5;

std::queue<int> buffer;
std::mutex mtx;
std::condition_variable not_full;
std::condition_variable not_empty;

void producer(int id) {
    for (int i = 0; i < 10; i++) {
        std::unique_lock<std::mutex> lock(mtx);

        not_full.wait(lock, []{ return buffer.size() < BUFFER_SIZE; });

        buffer.push(i);
        std::cout << "Producer " << id << ": produced " << i
                  << " (buffer size: " << buffer.size() << ")\n";

        not_empty.notify_one();
        lock.unlock();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void consumer(int id) {
    for (int i = 0; i < 10; i++) {
        std::unique_lock<std::mutex> lock(mtx);

        not_empty.wait(lock, []{ return !buffer.empty(); });

        int item = buffer.front();
        buffer.pop();
        std::cout << "Consumer " << id << ": consumed " << item
                  << " (buffer size: " << buffer.size() << ")\n";

        not_full.notify_one();
        lock.unlock();

        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

int main() {
    std::thread prod(producer, 1);
    std::thread cons(consumer, 1);

    prod.join();
    cons.join();

    return 0;
}
