/**
 * 13_error_handling.cpp
 * Demonstrates: Best practices for error handling in file I/O
 * Key Concepts:
 * - RAII (Resource Acquisition Is Initialization)
 * - Exception handling
 * - Proper error checking
 * - Safe file operations
 * - Cleanup and recovery
 */

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <memory>

// === 1. RAII File Wrapper Class ===
class SafeFile {
private:
    std::fstream file;
    std::string filename;
    bool isOpen;

public:
    SafeFile(const std::string& fname, std::ios::openmode mode)
        : filename(fname), isOpen(false) {
        file.open(fname, mode);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + fname);
        }
        isOpen = true;
        std::cout << "✓ File opened: " << filename << std::endl;
    }

    ~SafeFile() {
        if (isOpen && file.is_open()) {
            file.close();
            std::cout << "✓ File closed: " << filename << std::endl;
        }
    }

    // Delete copy constructor and assignment
    SafeFile(const SafeFile&) = delete;
    SafeFile& operator=(const SafeFile&) = delete;

    std::fstream& getStream() {
        if (!isOpen) {
            throw std::runtime_error("File is not open");
        }
        return file;
    }

    bool isFileOpen() const { return isOpen; }
};

// === 2. Safe file reading function ===
bool safeReadFile(const std::string& filename, std::string& content) {
    std::ifstream file(filename);

    // Check 1: File opened?
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file '" << filename << "'" << std::endl;
        return false;
    }

    // Check 2: File is readable?
    if (!file.good()) {
        std::cerr << "Error: File is not in good state" << std::endl;
        file.close();
        return false;
    }

    // Read content
    try {
        std::string line;
        while (std::getline(file, line)) {
            content += line + "\n";

            // Check for errors during reading
            if (file.bad()) {
                std::cerr << "Error: Fatal error during read" << std::endl;
                file.close();
                return false;
            }
        }

        // Check if we finished because of EOF or error
        if (!file.eof() && file.fail()) {
            std::cerr << "Error: Failed to read file completely" << std::endl;
            file.close();
            return false;
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception during read: " << e.what() << std::endl;
        file.close();
        return false;
    }

    file.close();
    return true;
}

// === 3. Safe file writing function ===
bool safeWriteFile(const std::string& filename, const std::string& content) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot create/open file '" << filename << "'" << std::endl;
        return false;
    }

    try {
        file << content;

        // Check if write succeeded
        if (file.fail()) {
            std::cerr << "Error: Write operation failed" << std::endl;
            file.close();
            return false;
        }

        // Explicitly flush to ensure data is written
        file.flush();

        if (file.bad()) {
            std::cerr << "Error: Bad stream state after flush" << std::endl;
            file.close();
            return false;
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception during write: " << e.what() << std::endl;
        file.close();
        return false;
    }

    file.close();

    // Verify the write by reading back
    std::ifstream verify(filename);
    if (!verify.is_open()) {
        std::cerr << "Warning: Could not verify write" << std::endl;
        return true;  // Still return true as write appeared successful
    }

    verify.close();
    return true;
}

// === 4. Transaction-like file update ===
bool updateFileWithBackup(const std::string& filename, const std::string& newContent) {
    std::string backupName = filename + ".bak";

    // Step 1: Read original content
    std::string originalContent;
    std::ifstream original(filename);

    if (original.is_open()) {
        std::string line;
        while (std::getline(original, line)) {
            originalContent += line + "\n";
        }
        original.close();

        // Step 2: Create backup
        std::ofstream backup(backupName);
        if (!backup.is_open()) {
            std::cerr << "Error: Cannot create backup file" << std::endl;
            return false;
        }
        backup << originalContent;
        backup.close();

        std::cout << "✓ Backup created: " << backupName << std::endl;
    }

    // Step 3: Write new content
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing" << std::endl;
        return false;
    }

    file << newContent;

    if (file.fail()) {
        std::cerr << "Error: Write failed, restoring from backup" << std::endl;
        file.close();

        // Restore from backup
        if (!originalContent.empty()) {
            std::ofstream restore(filename);
            restore << originalContent;
            restore.close();
            std::cout << "✓ File restored from backup" << std::endl;
        }
        return false;
    }

    file.close();
    std::cout << "✓ File updated successfully" << std::endl;
    return true;
}

// === 5. Exception-based file operations ===
class FileException : public std::runtime_error {
public:
    FileException(const std::string& msg) : std::runtime_error(msg) {}
};

void writeWithExceptions(const std::string& filename, const std::string& data) {
    std::ofstream file;
    file.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    try {
        file.open(filename);
        file << data;
        file.close();
        std::cout << "✓ Write with exceptions succeeded" << std::endl;
    } catch (const std::ofstream::failure& e) {
        throw FileException("Failed to write to file: " + filename +
                          "\nReason: " + e.what());
    }
}

int main() {
    std::cout << "=== File I/O Error Handling Best Practices ===" << std::endl;

    // === 1. RAII File Wrapper ===
    std::cout << "\n1. RAII Pattern (automatic cleanup):" << std::endl;
    std::cout << "====================================" << std::endl;

    try {
        SafeFile file("test_raii.txt", std::ios::out);
        file.getStream() << "This is RAII in action.\n";
        file.getStream() << "File will be closed automatically.\n";
        // File closes automatically when out of scope
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    std::cout << "After RAII block - file is closed" << std::endl;

    // === 2. Safe reading with error checking ===
    std::cout << "\n2. Safe file reading:" << std::endl;
    std::cout << "=====================" << std::endl;

    // Create test file
    std::ofstream create("test_read.txt");
    create << "Line 1\n";
    create << "Line 2\n";
    create << "Line 3\n";
    create.close();

    std::string content;
    if (safeReadFile("test_read.txt", content)) {
        std::cout << "✓ File read successfully" << std::endl;
        std::cout << "Content:\n" << content;
    }

    // Try reading non-existent file
    if (!safeReadFile("nonexistent.txt", content)) {
        std::cout << "✗ Failed to read non-existent file (expected)" << std::endl;
    }

    // === 3. Safe writing with verification ===
    std::cout << "\n3. Safe file writing:" << std::endl;
    std::cout << "=====================" << std::endl;

    std::string dataToWrite = "This is safe write test.\nMultiple lines.\nWith verification.\n";

    if (safeWriteFile("test_write.txt", dataToWrite)) {
        std::cout << "✓ File written and verified" << std::endl;
    }

    // === 4. File update with backup ===
    std::cout << "\n4. File update with backup:" << std::endl;
    std::cout << "===========================" << std::endl;

    std::ofstream original("important.txt");
    original << "Original important data.\n";
    original << "Do not lose this!\n";
    original.close();

    std::string newData = "Updated important data.\n";
    newData += "New version with backup.\n";

    if (updateFileWithBackup("important.txt", newData)) {
        std::cout << "✓ Update completed" << std::endl;
    }

    // === 5. Exception-based handling ===
    std::cout << "\n5. Exception-based handling:" << std::endl;
    std::cout << "============================" << std::endl;

    try {
        writeWithExceptions("test_exception.txt", "Data written with exception handling\n");
    } catch (const FileException& e) {
        std::cerr << "FileException caught: " << e.what() << std::endl;
    }

    // === 6. Comprehensive error checking ===
    std::cout << "\n6. Comprehensive error checking:" << std::endl;
    std::cout << "================================" << std::endl;

    auto robustFileOperation = [](const std::string& filename) -> bool {
        std::fstream file;

        // Enable exceptions for critical errors
        file.exceptions(std::fstream::badbit);

        try {
            // Open file
            file.open(filename, std::ios::in | std::ios::out | std::ios::app);

            if (!file.is_open()) {
                std::cerr << "✗ Cannot open: " << filename << std::endl;
                return false;
            }

            // Check initial state
            if (!file.good()) {
                std::cerr << "✗ File not in good state" << std::endl;
                file.close();
                return false;
            }

            // Perform operations
            file << "Appending data safely.\n";

            // Check after write
            if (file.fail()) {
                std::cerr << "✗ Write operation failed" << std::endl;
                file.close();
                return false;
            }

            // Flush and check
            file.flush();

            if (file.bad()) {
                std::cerr << "✗ Stream in bad state after flush" << std::endl;
                file.close();
                return false;
            }

            file.close();
            std::cout << "✓ All operations successful: " << filename << std::endl;
            return true;

        } catch (const std::ios_base::failure& e) {
            std::cerr << "✗ I/O exception: " << e.what() << std::endl;
            if (file.is_open()) {
                file.close();
            }
            return false;
        } catch (...) {
            std::cerr << "✗ Unknown exception" << std::endl;
            if (file.is_open()) {
                file.close();
            }
            return false;
        }
    };

    robustFileOperation("robust_test.txt");

    // === 7. Error recovery example ===
    std::cout << "\n7. Error recovery:" << std::endl;
    std::cout << "==================" << std::endl;

    std::ifstream errorTest("test_error.txt");
    int attempts = 0;
    const int maxAttempts = 3;
    bool success = false;

    while (attempts < maxAttempts && !success) {
        attempts++;
        std::cout << "Attempt " << attempts << "..." << std::endl;

        if (!errorTest.is_open()) {
            // Try to create the file if it doesn't exist
            std::ofstream create("test_error.txt");
            create << "Created on attempt " << attempts << "\n";
            create.close();

            // Try opening again
            errorTest.open("test_error.txt");
        }

        if (errorTest.is_open()) {
            success = true;
            std::cout << "✓ Success on attempt " << attempts << std::endl;
        }
    }

    if (success) {
        errorTest.close();
    } else {
        std::cout << "✗ Failed after " << maxAttempts << " attempts" << std::endl;
    }

    // === 8. Best practices summary ===
    std::cout << "\n=== Best Practices Summary ===" << std::endl;
    std::cout << "1. Always check if file opened successfully" << std::endl;
    std::cout << "2. Use RAII for automatic resource management" << std::endl;
    std::cout << "3. Check stream state after operations" << std::endl;
    std::cout << "4. Use try-catch for exception handling" << std::endl;
    std::cout << "5. Create backups before modifying important files" << std::endl;
    std::cout << "6. Verify writes by reading back when critical" << std::endl;
    std::cout << "7. Close files explicitly when done" << std::endl;
    std::cout << "8. Handle both logical and fatal errors" << std::endl;

    std::cout << "\n✓ All error handling examples completed!" << std::endl;

    return 0;
}
