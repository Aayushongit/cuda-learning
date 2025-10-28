/**
 * 03_append_mode.cpp
 * Demonstrates: Appending data to existing files
 * Key Concepts:
 * - std::ios::app mode
 * - Difference between append and write modes
 * - Preserving existing content
 * - Logging pattern
 */

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

std::string getCurrentTime() {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

int main() {
    std::cout << "=== File Append Operations ===" << std::endl;

    // First, create a file with initial content
    std::ofstream initial("log.txt");
    initial << "=== Application Log ===" << std::endl;
    initial << "Started at: " << getCurrentTime() << std::endl;
    initial.close();

    std::cout << "Initial log file created." << std::endl;

    // Append mode: std::ios::app
    // This opens file for writing but preserves existing content
    std::ofstream logfile("log.txt", std::ios::app);

    if (!logfile.is_open()) {
        std::cerr << "Error: Could not open log file!" << std::endl;
        return 1;
    }

    // Simulate logging events
    std::cout << "\nAppending log entries..." << std::endl;
    logfile << "[INFO] " << getCurrentTime() << " - System initialized" << std::endl;
    logfile << "[INFO] " << getCurrentTime() << " - Loading configuration" << std::endl;
    logfile << "[WARN] " << getCurrentTime() << " - Low memory warning" << std::endl;
    logfile << "[INFO] " << getCurrentTime() << " - Processing data" << std::endl;

    logfile.close();

    // Append more entries in a separate session
    std::ofstream logfile2("log.txt", std::ios::app);
    logfile2 << "[INFO] " << getCurrentTime() << " - Task completed" << std::endl;
    logfile2 << "[INFO] " << getCurrentTime() << " - Shutting down" << std::endl;
    logfile2.close();

    // Demonstrate the difference: Compare append vs. overwrite
    std::cout << "\nDemonstrating append vs. overwrite:" << std::endl;

    // Create test file
    std::ofstream test("test_modes.txt");
    test << "Original line 1\n";
    test << "Original line 2\n";
    test.close();

    // Append mode - preserves content
    std::ofstream append_test("test_modes.txt", std::ios::app);
    append_test << "Appended line 3\n";
    append_test.close();

    // Show the result
    std::cout << "\nFinal log.txt content:" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::ifstream read_log("log.txt");
    std::string line;
    while (std::getline(read_log, line)) {
        std::cout << line << std::endl;
    }
    read_log.close();

    std::cout << "\ntest_modes.txt content (with append):" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    std::ifstream read_test("test_modes.txt");
    while (std::getline(read_test, line)) {
        std::cout << line << std::endl;
    }
    read_test.close();

    // Now show what happens with overwrite
    std::ofstream overwrite_test("test_modes.txt"); // No append flag
    overwrite_test << "This overwrites everything!\n";
    overwrite_test.close();

    std::cout << "\ntest_modes.txt after overwrite:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    std::ifstream read_overwrite("test_modes.txt");
    while (std::getline(read_overwrite, line)) {
        std::cout << line << std::endl;
    }
    read_overwrite.close();

    return 0;
}
