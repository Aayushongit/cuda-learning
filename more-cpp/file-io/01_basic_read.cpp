/**
 * 01_basic_read.cpp
 * Demonstrates: Basic file reading using ifstream
 * Key Concepts:
 * - Opening files for reading
 * - Checking if file opened successfully
 * - Reading entire file content
 * - Proper file closure
 */

#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::cout << "=== Basic File Reading ===" << std::endl;

    // Create a test file first
    std::ofstream create_file("sample.txt");
    create_file << "Hello, File I/O!\n";
    create_file << "This is line 2.\n";
    create_file << "Learning C++ file operations.";
    create_file.close();

    // Method 1: Read entire file into string
    std::ifstream file("sample.txt");

    // Always check if file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file!" << std::endl;
        return 1;
    }

    std::cout << "\nMethod 1: Reading character by character:" << std::endl;
    char ch;
    while (file.get(ch)) {
        std::cout << ch;
    }

    // Close and reopen for next method
    file.close();

    // Method 2: Read word by word
    file.open("sample.txt");
    std::cout << "\n\nMethod 2: Reading word by word:" << std::endl;
    std::string word;
    while (file >> word) {
        std::cout << word << " | ";
    }

    file.close();

    // Method 3: Read entire file at once
    file.open("sample.txt");
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    std::cout << "\n\nMethod 3: Entire file content:\n" << content << std::endl;

    file.close();

    std::cout << "\nFile closed successfully!" << std::endl;

    return 0;
}
