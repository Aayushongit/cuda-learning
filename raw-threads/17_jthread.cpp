#include <iostream>
#include <thread>
#include <chrono>

void cancelable_worker(std::stop_token stoken, int id) {
    for (int i = 0; i < 10; i++) {
        if (stoken.stop_requested()) {
            std::cout << "Worker " << id << ": Stop requested, cleaning up\n";
            return;
        }

        std::cout << "Worker " << id << ": Iteration " << i << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

int main() {
    std::cout << "Creating jthread (auto-joining, cancelable)\n";
    std::jthread worker(cancelable_worker, 1);

    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Main: Requesting stop\n";
    worker.request_stop();

    std::cout << "Main: jthread will auto-join on destruction\n";

    return 0;
}
