/**
 * 05_line_by_line.cpp
 * Demonstrates: Line-by-line file reading
 * Key Concepts:
 * - getline() function
 * - Processing text files line by line
 * - Handling different line endings
 * - Practical text processing
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main() {
    std::cout << "=== Line-by-Line File Reading ===" << std::endl;

    // Create a sample text file
    std::ofstream create("poem.txt");
    create << "The Road Not Taken\n";
    create << "By Robert Frost\n";
    create << "\n";
    create << "Two roads diverged in a yellow wood,\n";
    create << "And sorry I could not travel both\n";
    create << "And be one traveler, long I stood\n";
    create << "And looked down one as far as I could\n";
    create << "To where it bent in the undergrowth;\n";
    create.close();

    // === Method 1: Basic getline() ===
    std::cout << "\n1. Basic line-by-line reading:" << std::endl;
    std::cout << "================================" << std::endl;

    std::ifstream file("poem.txt");
    std::string line;
    int lineNumber = 1;

    while (std::getline(file, line)) {
        std::cout << lineNumber << ": " << line << std::endl;
        lineNumber++;
    }
    file.close();

    // === Method 2: Store lines in vector ===
    std::cout << "\n2. Storing lines in a vector:" << std::endl;
    std::cout << "==============================" << std::endl;

    file.open("poem.txt");
    std::vector<std::string> lines;

    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    std::cout << "Total lines read: " << lines.size() << std::endl;
    std::cout << "First line: " << lines[0] << std::endl;
    std::cout << "Last line: " << lines[lines.size() - 1] << std::endl;

    // === Method 3: Processing lines (counting words) ===
    std::cout << "\n3. Processing each line (word count):" << std::endl;
    std::cout << "=====================================" << std::endl;

    file.open("poem.txt");
    int totalWords = 0;
    lineNumber = 1;

    while (std::getline(file, line)) {
        if (line.empty()) {
            std::cout << "Line " << lineNumber << ": [empty line]" << std::endl;
        } else {
            int wordCount = 0;
            bool inWord = false;

            for (char c : line) {
                if (std::isspace(c)) {
                    inWord = false;
                } else if (!inWord) {
                    inWord = true;
                    wordCount++;
                }
            }

            totalWords += wordCount;
            std::cout << "Line " << lineNumber << ": " << wordCount << " words" << std::endl;
        }
        lineNumber++;
    }
    file.close();

    std::cout << "\nTotal words in file: " << totalWords << std::endl;

    // === Method 4: Filtering lines ===
    std::cout << "\n4. Filtering lines (containing specific text):" << std::endl;
    std::cout << "=============================================" << std::endl;

    file.open("poem.txt");
    std::string searchTerm = "road";

    std::cout << "Lines containing '" << searchTerm << "':" << std::endl;
    lineNumber = 1;

    while (std::getline(file, line)) {
        // Convert line to lowercase for case-insensitive search
        std::string lowerLine = line;
        for (char& c : lowerLine) {
            c = std::tolower(c);
        }

        if (lowerLine.find(searchTerm) != std::string::npos) {
            std::cout << "  Line " << lineNumber << ": " << line << std::endl;
        }
        lineNumber++;
    }
    file.close();

    // === Method 5: Reading with custom delimiter ===
    std::cout << "\n5. Custom delimiter (reading by sentence):" << std::endl;
    std::cout << "==========================================" << std::endl;

    // Create a file with sentences
    std::ofstream sentences_file("sentences.txt");
    sentences_file << "First sentence. Second sentence. Third sentence.";
    sentences_file.close();

    std::ifstream sent_in("sentences.txt");
    std::string sentence;
    int sentNum = 1;

    while (std::getline(sent_in, sentence, '.')) {
        if (!sentence.empty()) {
            std::cout << "Sentence " << sentNum << ": " << sentence << std::endl;
            sentNum++;
        }
    }
    sent_in.close();

    // === Method 6: Reverse file reading (read into vector then reverse) ===
    std::cout << "\n6. Reading file in reverse order:" << std::endl;
    std::cout << "==================================" << std::endl;

    file.open("poem.txt");
    lines.clear();

    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    for (int i = lines.size() - 1; i >= 0; i--) {
        std::cout << lines[i] << std::endl;
    }

    return 0;
}
