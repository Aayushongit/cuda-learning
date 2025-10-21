#include <iostream>
#include <thread>

void worker() {
    std::cout << "Thread ID: " << std::this_thread::get_id() << "\n";
}

int main() {
    unsigned int num_cores = std::thread::hardware_concurrency();

    std::cout << "Hardware concurrency: " << num_cores << " cores\n";
    std::cout << "Recommended thread count for CPU-bound tasks: " << num_cores << "\n";
    std::cout << "Main thread ID: " << std::this_thread::get_id() << "\n\n";

    std::thread t1(worker);
    std::thread t2(worker);

    std::cout << "Created thread 1: " << t1.get_id() << "\n";
    std::cout << "Created thread 2: " << t2.get_id() << "\n";

    t1.join();
    t2.join();

    return 0;
}
