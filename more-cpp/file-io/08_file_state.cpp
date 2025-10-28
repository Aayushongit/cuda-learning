/**
 * 08_file_state.cpp
 * Demonstrates: Checking file states and error handling
 * Key Concepts:
 * - is_open() - check if file opened successfully
 * - good() - stream is ready for I/O
 * - eof() - end of file reached
 * - fail() - logical error occurred
 * - bad() - read/write error occurred
 * - clear() - clear error flags
 */

#include <iostream>
#include <fstream>
#include <string>

void printStreamState(std::ios& stream, const std::string& context) {
    std::cout << "\n[" << context << "]" << std::endl;
    std::cout << "  good():  " << stream.good() << "  (ready for I/O)" << std::endl;
    std::cout << "  eof():   " << stream.eof() << "   (end of file)" << std::endl;
    std::cout << "  fail():  " << stream.fail() << "   (logical error)" << std::endl;
    std::cout << "  bad():   " << stream.bad() << "   (read/write error)" << std::endl;
}

int main() {
    std::cout << "=== File State Checking ===" << std::endl;

    // === 1. is_open() - Checking if file opened ===
    std::cout << "\n1. Checking if file opened:" << std::endl;
    std::cout << "===========================" << std::endl;

    std::ifstream file1("nonexistent_file.txt");

    if (!file1.is_open()) {
        std::cout << "✗ File did not open (doesn't exist)" << std::endl;
    } else {
        std::cout << "✓ File opened successfully" << std::endl;
    }

    // Create and open existing file
    std::ofstream create("test.txt");
    create << "This file exists!";
    create.close();

    std::ifstream file2("test.txt");
    if (file2.is_open()) {
        std::cout << "✓ File opened successfully (exists)" << std::endl;
        file2.close();
    }

    // === 2. good() - Stream is ready ===
    std::cout << "\n2. Stream state with good():" << std::endl;
    std::cout << "============================" << std::endl;

    std::ifstream file3("test.txt");
    printStreamState(file3, "After opening");

    char ch;
    while (file3.get(ch)) {
        // Reading...
    }
    printStreamState(file3, "After reading to end");

    file3.close();

    // === 3. eof() - End of file detection ===
    std::cout << "\n3. End of file detection:" << std::endl;
    std::cout << "=========================" << std::endl;

    std::ofstream createLines("lines.txt");
    createLines << "Line 1\n";
    createLines << "Line 2\n";
    createLines << "Line 3\n";
    createLines.close();

    std::ifstream file4("lines.txt");
    std::string line;
    int lineNum = 0;

    while (std::getline(file4, line)) {
        lineNum++;
        std::cout << "Read line " << lineNum << ": " << line << std::endl;
        std::cout << "  eof() = " << file4.eof() << std::endl;
    }

    std::cout << "\nAfter loop:" << std::endl;
    std::cout << "  eof() = " << file4.eof() << std::endl;
    std::cout << "  good() = " << file4.good() << std::endl;

    file4.close();

    // === 4. fail() - Logical errors ===
    std::cout << "\n4. Logical error detection (fail):" << std::endl;
    std::cout << "==================================" << std::endl;

    std::ofstream createNum("number.txt");
    createNum << "42 hello 99";
    createNum.close();

    std::ifstream file5("number.txt");
    int num;

    file5 >> num;
    std::cout << "Read number: " << num << ", fail() = " << file5.fail() << std::endl;

    file5 >> num;  // Try to read "hello" as number
    std::cout << "Try to read 'hello' as number, fail() = " << file5.fail() << std::endl;

    printStreamState(file5, "After failed read");

    file5.close();

    // === 5. clear() - Clearing error states ===
    std::cout << "\n5. Clearing error states:" << std::endl;
    std::cout << "=========================" << std::endl;

    std::ifstream file6("number.txt");

    file6 >> num;
    std::cout << "Read: " << num << std::endl;

    file6 >> num;  // This will fail
    std::cout << "Failed read, fail() = " << file6.fail() << std::endl;

    // Try to read more - won't work until we clear
    file6 >> num;
    std::cout << "Try again without clear, fail() = " << file6.fail() << std::endl;

    // Clear the error state
    file6.clear();
    std::cout << "After clear(), fail() = " << file6.fail() << std::endl;

    // Now we can continue (though we need to skip the bad input)
    file6.ignore(100, ' ');  // Skip to next space
    file6 >> num;
    std::cout << "After clear and ignore, read: " << num << std::endl;

    file6.close();

    // === 6. Checking file before operations ===
    std::cout << "\n6. Safe file operations pattern:" << std::endl;
    std::cout << "================================" << std::endl;

    std::ifstream safeFile("test.txt");

    // Check if opened
    if (!safeFile.is_open()) {
        std::cerr << "✗ Error: Could not open file!" << std::endl;
        return 1;
    }

    std::cout << "✓ File opened" << std::endl;

    // Check state before reading
    if (safeFile.good()) {
        std::cout << "✓ Stream is good, proceeding to read" << std::endl;

        std::string content;
        while (std::getline(safeFile, line) && safeFile.good()) {
            content += line + "\n";
        }

        if (safeFile.eof()) {
            std::cout << "✓ Reached end of file normally" << std::endl;
        } else if (safeFile.fail()) {
            std::cout << "✗ Error during reading" << std::endl;
        }
    }

    safeFile.close();

    // === 7. Exception handling with files ===
    std::cout << "\n7. Using exceptions with files:" << std::endl;
    std::cout << "===============================" << std::endl;

    std::ifstream file7;
    file7.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
        file7.open("test.txt");
        std::cout << "✓ File opened with exceptions enabled" << std::endl;

        std::string content;
        while (std::getline(file7, line)) {
            content += line;
        }

        file7.close();
        std::cout << "✓ Read completed successfully" << std::endl;

    } catch (std::ifstream::failure& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }

    // === 8. Complete error checking function ===
    std::cout << "\n8. Complete error checking example:" << std::endl;
    std::cout << "===================================" << std::endl;

    auto checkAndRead = [](const std::string& filename) {
        std::ifstream file(filename);

        // Check 1: File opened?
        if (!file.is_open()) {
            std::cerr << "✗ Cannot open file: " << filename << std::endl;
            return false;
        }

        // Check 2: Stream is good?
        if (!file.good()) {
            std::cerr << "✗ Stream not in good state" << std::endl;
            file.close();
            return false;
        }

        // Read file
        std::string line;
        bool success = true;

        while (std::getline(file, line)) {
            if (file.bad()) {
                std::cerr << "✗ Fatal error during read" << std::endl;
                success = false;
                break;
            }
        }

        // Check 3: Did we finish because of EOF or error?
        if (success && file.eof()) {
            std::cout << "✓ File read successfully: " << filename << std::endl;
        } else if (file.fail() && !file.eof()) {
            std::cerr << "✗ Error reading file" << std::endl;
            success = false;
        }

        file.close();
        return success;
    };

    checkAndRead("test.txt");
    checkAndRead("nonexistent.txt");

    std::cout << "\n✓ All state checking examples completed!" << std::endl;

    return 0;
}
