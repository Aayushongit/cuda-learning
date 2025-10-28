/**
 * 02_basic_write.cpp
 * Demonstrates: Basic file writing using ofstream
 * Key Concepts:
 * - Opening files for writing
 * - Different write modes
 * - Writing different data types
 * - Overwriting vs. creating new files
 */

#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::cout << "=== Basic File Writing ===" << std::endl;

    // Method 1: Simple write (overwrites if exists)
    std::ofstream outfile("output.txt");

    if (!outfile.is_open()) {
        std::cerr << "Error: Could not create/open file!" << std::endl;
        return 1;
    }

    // Writing different data types
    outfile << "Writing to file in C++" << std::endl;
    outfile << "Integer: " << 42 << std::endl;
    outfile << "Float: " << 3.14159 << std::endl;
    outfile << "Boolean: " << std::boolalpha << true << std::endl;

    outfile.close();
    std::cout << "Data written to output.txt" << std::endl;

    // Method 2: Using std::ios modes explicitly
    // std::ios::out - open for writing (default for ofstream)
    // std::ios::trunc - discard contents if file exists (default)
    std::ofstream file2("numbers.txt", std::ios::out | std::ios::trunc);

    std::cout << "\nWriting numbers to numbers.txt..." << std::endl;
    for (int i = 1; i <= 10; i++) {
        file2 << "Number " << i << ": " << i * i << std::endl;
    }
    file2.close();

    // Method 3: Writing formatted data
    std::ofstream data_file("formatted_data.txt");

    data_file << "Name\t\tAge\tScore" << std::endl;
    data_file << "----------------------------------------" << std::endl;
    data_file << "Alice\t\t25\t92.5" << std::endl;
    data_file << "Bob\t\t30\t87.3" << std::endl;
    data_file << "Charlie\t\t28\t95.8" << std::endl;

    data_file.close();
    std::cout << "Formatted data written to formatted_data.txt" << std::endl;

    // Verify by reading back
    std::ifstream verify("output.txt");
    std::string line;
    std::cout << "\nVerifying output.txt content:" << std::endl;
    std::cout << "------------------------------" << std::endl;
    while (std::getline(verify, line)) {
        std::cout << line << std::endl;
    }
    verify.close();

    return 0;
}
