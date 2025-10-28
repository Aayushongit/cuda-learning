// Basic Thread Creation and Management
// This file demonstrates how to create and join threads in C++

#include <iostream>
#include <thread>
#include <chrono>

// Simple function to run in a thread
void printNumbers(int count) {
    for (int i = 1; i <= count; i++) {
        std::cout << "Thread 1: " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Function with multiple parameters
void printMessage(const std::string& msg, int times) {
    for (int i = 0; i < times; i++) {
        std::cout << msg << " (" << i + 1 << ")" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

// Lambda function example
void lambdaThreadExample() {
    std::thread t([]() {
        std::cout << "Lambda thread running!" << std::endl;
    });
    t.join();
}

int main() {
    std::cout << "=== Basic Thread Creation ===" << std::endl;

    // Create thread with function and argument
    std::thread thread1(printNumbers, 5);

    // Create thread with multiple arguments
    std::thread thread2(printMessage, "Hello from thread 2", 3);

    // Main thread also does work
    for (int i = 0; i < 3; i++) {
        std::cout << "Main thread working..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Wait for threads to complete (join them back to main)
    thread1.join();
    thread2.join();

    std::cout << "\n=== Lambda Thread Example ===" << std::endl;
    lambdaThreadExample();

    std::cout << "\nAll threads completed!" << std::endl;
    return 0;
}
