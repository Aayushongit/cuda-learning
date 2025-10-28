// Promise and Future
// Promise allows setting a value that future can retrieve

#include <iostream>
#include <thread>
#include <future>
#include <chrono>

// Promise allows one thread to send data to another
void producer(std::promise<int>&& promise) {
    std::cout << "Producer: Starting work..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    int result = 42;
    std::cout << "Producer: Setting value " << result << std::endl;

    // Set the value (consumer's future will receive it)
    promise.set_value(result);
}

void consumer(std::future<int>&& future) {
    std::cout << "Consumer: Waiting for data..." << std::endl;

    // Wait for producer to set value
    int value = future.get();

    std::cout << "Consumer: Received value " << value << std::endl;
}

// Promise with exception
void producerWithException(std::promise<int>&& promise, bool shouldFail) {
    try {
        if (shouldFail) {
            throw std::runtime_error("Something went wrong!");
        }

        promise.set_value(100);
    } catch (...) {
        // Pass exception to future
        promise.set_exception(std::current_exception());
    }
}

// Multiple values using promise
void multiValueExample() {
    std::promise<std::string> promise;
    std::future<std::string> future = promise.get_future();

    std::thread worker([](std::promise<std::string> p) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        p.set_value("Hello from worker thread!");
    }, std::move(promise));

    std::cout << "Waiting for worker..." << std::endl;
    std::string message = future.get();
    std::cout << "Received: " << message << std::endl;

    worker.join();
}

// Using promise for synchronization
void synchronizationExample() {
    std::promise<void> startSignal;
    std::future<void> ready = startSignal.get_future();

    std::thread worker([](std::future<void> fut) {
        std::cout << "Worker: Waiting for start signal..." << std::endl;
        fut.wait();  // Wait for signal
        std::cout << "Worker: Started working!" << std::endl;
    }, std::move(ready));

    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Main: Sending start signal" << std::endl;
    startSignal.set_value();  // Signal worker to start

    worker.join();
}

int main() {
    std::cout << "=== Promise and Future ===" << std::endl;

    std::promise<int> prom;
    std::future<int> fut = prom.get_future();

    std::thread producerThread(producer, std::move(prom));
    std::thread consumerThread(consumer, std::move(fut));

    producerThread.join();
    consumerThread.join();

    std::cout << "\n=== Promise with Exception ===" << std::endl;

    std::promise<int> prom2;
    std::future<int> fut2 = prom2.get_future();

    std::thread t(producerWithException, std::move(prom2), true);

    try {
        int value = fut2.get();
        std::cout << "Received: " << value << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }

    t.join();

    std::cout << "\n=== Multiple Values ===" << std::endl;
    multiValueExample();

    std::cout << "\n=== Synchronization with Promise ===" << std::endl;
    synchronizationExample();

    return 0;
}
