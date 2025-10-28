/**
 * 11_file_utilities.cpp
 * Demonstrates: File manipulation utilities
 * Key Concepts:
 * - Checking file existence
 * - Getting file size
 * - Copying files
 * - File comparison
 * - Temporary files
 * - File operations using C++17 filesystem (bonus)
 */

#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <ctime>
#include <vector>


// Check if file exists
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// Get file size
long getFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        return -1;
    }
    return static_cast<long>(file.tellg());
}

// Copy file
bool copyFile(const std::string& source, const std::string& destination) {
    std::ifstream src(source, std::ios::binary);
    if (!src) {
        return false;
    }

    std::ofstream dst(destination, std::ios::binary);
    if (!dst) {
        return false;
    }

    // Copy byte by byte
    dst << src.rdbuf();

    return src.good() && dst.good();
}

// Compare two files
bool filesAreIdentical(const std::string& file1, const std::string& file2) {
    std::ifstream f1(file1, std::ios::binary);
    std::ifstream f2(file2, std::ios::binary);

    if (!f1 || !f2) {
        return false;
    }

    // Compare sizes first
    f1.seekg(0, std::ios::end);
    f2.seekg(0, std::ios::end);

    if (f1.tellg() != f2.tellg()) {
        return false;
    }

    // Compare byte by byte
    f1.seekg(0, std::ios::beg);
    f2.seekg(0, std::ios::beg);

    char ch1, ch2;
    while (f1.get(ch1) && f2.get(ch2)) {
        if (ch1 != ch2) {
            return false;
        }
    }

    return true;
}

// Count lines in file
int countLines(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        return -1;
    }

    int count = 0;
    std::string line;
    while (std::getline(file, line)) {
        count++;
    }

    return count;
}

// Count words in file
int countWords(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        return -1;
    }

    int count = 0;
    std::string word;
    while (file >> word) {
        count++;
    }

    return count;
}

// Get file modification time
std::string getFileModificationTime(const std::string& filename) {
    struct stat fileInfo;
    if (stat(filename.c_str(), &fileInfo) != 0) {
        return "Unknown";
    }

    char buffer[80];
    struct tm* timeinfo = localtime(&fileInfo.st_mtime);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);

    return std::string(buffer);
}

// Read last N lines of file
void readLastNLines(const std::string& filename, int n) {
    std::ifstream file(filename);
    if (!file) {
        std::cout << "Cannot open file!" << std::endl;
        return;
    }

    // Read all lines into vector
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    // Print last N lines
    int start = std::max(0, static_cast<int>(lines.size()) - n);
    for (size_t i = start; i < lines.size(); i++) {
        std::cout << lines[i] << std::endl;
    }
}

// Merge two files
bool mergeFiles(const std::string& file1, const std::string& file2,
                const std::string& output) {
    std::ofstream out(output);
    if (!out) {
        return false;
    }

    // Copy first file
    std::ifstream in1(file1);
    if (in1) {
        out << in1.rdbuf();
        in1.close();
    }

    // Copy second file
    std::ifstream in2(file2);
    if (in2) {
        out << in2.rdbuf();
        in2.close();
    }

    out.close();
    return true;
}

int main() {
    std::cout << "=== File Manipulation Utilities ===" << std::endl;

    // === 1. Check file existence ===
    std::cout << "\n1. Checking file existence:" << std::endl;
    std::cout << "===========================" << std::endl;

    // Create a test file
    std::ofstream create("test_util.txt");
    create << "This is a test file for utilities.\n";
    create << "It has multiple lines.\n";
    create << "We will perform various operations on it.\n";
    create.close();

    std::cout << "test_util.txt exists: " << (fileExists("test_util.txt") ? "Yes" : "No") << std::endl;
    std::cout << "nonexistent.txt exists: " << (fileExists("nonexistent.txt") ? "Yes" : "No") << std::endl;

    // === 2. Get file size ===
    std::cout << "\n2. Getting file size:" << std::endl;
    std::cout << "=====================" << std::endl;

    long size = getFileSize("test_util.txt");
    std::cout << "test_util.txt size: " << size << " bytes" << std::endl;

    // === 3. Copy file ===
    std::cout << "\n3. Copying file:" << std::endl;
    std::cout << "================" << std::endl;

    if (copyFile("test_util.txt", "test_util_copy.txt")) {
        std::cout << "✓ File copied successfully" << std::endl;
        std::cout << "Copy size: " << getFileSize("test_util_copy.txt") << " bytes" << std::endl;
    }

    // === 4. Compare files ===
    std::cout << "\n4. Comparing files:" << std::endl;
    std::cout << "===================" << std::endl;

    bool identical = filesAreIdentical("test_util.txt", "test_util_copy.txt");
    std::cout << "test_util.txt and test_util_copy.txt are "
              << (identical ? "identical" : "different") << std::endl;

    // Create a different file
    std::ofstream different("different.txt");
    different << "This is different content.\n";
    different.close();

    identical = filesAreIdentical("test_util.txt", "different.txt");
    std::cout << "test_util.txt and different.txt are "
              << (identical ? "identical" : "different") << std::endl;

    // === 5. Count lines and words ===
    std::cout << "\n5. Counting lines and words:" << std::endl;
    std::cout << "============================" << std::endl;

    int lines = countLines("test_util.txt");
    int words = countWords("test_util.txt");

    std::cout << "test_util.txt statistics:" << std::endl;
    std::cout << "  Lines: " << lines << std::endl;
    std::cout << "  Words: " << words << std::endl;
    std::cout << "  Bytes: " << getFileSize("test_util.txt") << std::endl;

    // === 6. File modification time ===
    std::cout << "\n6. File modification time:" << std::endl;
    std::cout << "==========================" << std::endl;

    std::string modTime = getFileModificationTime("test_util.txt");
    std::cout << "test_util.txt last modified: " << modTime << std::endl;

    // === 7. Read last N lines ===
    std::cout << "\n7. Reading last 2 lines:" << std::endl;
    std::cout << "========================" << std::endl;

    readLastNLines("test_util.txt", 2);

    // === 8. Merge files ===
    std::cout << "\n8. Merging files:" << std::endl;
    std::cout << "=================" << std::endl;

    std::ofstream file1("part1.txt");
    file1 << "This is part 1.\n";
    file1 << "Content from first file.\n";
    file1.close();

    std::ofstream file2("part2.txt");
    file2 << "This is part 2.\n";
    file2 << "Content from second file.\n";
    file2.close();

    if (mergeFiles("part1.txt", "part2.txt", "merged.txt")) {
        std::cout << "✓ Files merged successfully" << std::endl;
        std::cout << "Merged file size: " << getFileSize("merged.txt") << " bytes" << std::endl;

        std::cout << "\nMerged content:" << std::endl;
        std::ifstream merged("merged.txt");
        std::string line;
        while (std::getline(merged, line)) {
            std::cout << "  " << line << std::endl;
        }
        merged.close();
    }

    // === 9. File information summary ===
    std::cout << "\n9. File information summary:" << std::endl;
    std::cout << "============================" << std::endl;

    auto printFileInfo = [](const std::string& filename) {
        std::cout << "\nFile: " << filename << std::endl;
        std::cout << "  Exists: " << (fileExists(filename) ? "Yes" : "No") << std::endl;
        if (fileExists(filename)) {
            std::cout << "  Size: " << getFileSize(filename) << " bytes" << std::endl;
            std::cout << "  Lines: " << countLines(filename) << std::endl;
            std::cout << "  Words: " << countWords(filename) << std::endl;
            std::cout << "  Modified: " << getFileModificationTime(filename) << std::endl;
        }
    };

    printFileInfo("test_util.txt");
    printFileInfo("merged.txt");
    printFileInfo("different.txt");

    // === 10. Temporary file pattern ===
    std::cout << "\n10. Creating temporary file:" << std::endl;
    std::cout << "============================" << std::endl;

    // Generate temporary filename
    std::string tempFilename = "temp_" + std::to_string(std::time(nullptr)) + ".tmp";

    std::ofstream tempFile(tempFilename);
    tempFile << "This is temporary data.\n";
    tempFile << "It will be processed and deleted.\n";
    tempFile.close();

    std::cout << "✓ Created temporary file: " << tempFilename << std::endl;
    std::cout << "Size: " << getFileSize(tempFilename) << " bytes" << std::endl;

    // Process the temp file...
    int lineCount = countLines(tempFilename);
    std::cout << "Processed " << lineCount << " lines" << std::endl;

    // Delete temp file (in real code)
    // std::remove(tempFilename.c_str());

    std::cout << "\n✓ All file utility examples completed!" << std::endl;
    std::cout << "\nFiles created:" << std::endl;
    std::cout << "  - test_util.txt" << std::endl;
    std::cout << "  - test_util_copy.txt" << std::endl;
    std::cout << "  - different.txt" << std::endl;
    std::cout << "  - part1.txt, part2.txt" << std::endl;
    std::cout << "  - merged.txt" << std::endl;
    std::cout << "  - " << tempFilename << std::endl;

    return 0;
}
