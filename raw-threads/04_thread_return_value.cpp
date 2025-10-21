#include <iostream>
#include <thread>
#include <future>

int calculate_square(int num) {
    return num * num;
}

int main() {
    int number = 7;

    std::future<int> result = std::async(std::launch::async, calculate_square, number);

    std::cout << "Square of " << number << " is " << result.get() << "\n";

    return 0;
}
