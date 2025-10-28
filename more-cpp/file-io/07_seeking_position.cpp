/**
 * 07_seeking_position.cpp
 * Demonstrates: File positioning and seeking
 * Key Concepts:
 * - tellg() and tellp() - get current position
 * - seekg() and seekp() - set position
 * - std::ios::beg, std::ios::cur, std::ios::end
 * - Random access to file content
 */

#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::cout << "=== File Positioning and Seeking ===" << std::endl;

    // Create a sample file
    std::ofstream create("positions.txt");
    create << "0123456789\n";          // Line 0: chars 0-10
    create << "ABCDEFGHIJ\n";          // Line 1: chars 11-21
    create << "abcdefghij\n";          // Line 2: chars 22-32
    create.close();

    // === 1. tellg() - Get current read position ===
    std::cout << "\n1. Getting current position (tellg):" << std::endl;
    std::cout << "====================================" << std::endl;

    std::ifstream file("positions.txt");
    char ch;

    std::cout << "Initial position: " << file.tellg() << std::endl;

    file.get(ch);
    std::cout << "Read char '" << ch << "', position now: " << file.tellg() << std::endl;

    file.get(ch);
    std::cout << "Read char '" << ch << "', position now: " << file.tellg() << std::endl;

    // === 2. seekg() - Set read position ===
    std::cout << "\n2. Seeking to specific positions (seekg):" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Seek to beginning
    file.seekg(0, std::ios::beg);
    file.get(ch);
    std::cout << "Seeked to beginning (0), read: '" << ch << "'" << std::endl;

    // Seek to position 5 from beginning
    file.seekg(5, std::ios::beg);
    file.get(ch);
    std::cout << "Seeked to position 5, read: '" << ch << "'" << std::endl;

    // Seek to end of file
    file.seekg(0, std::ios::end);
    std::cout << "Seeked to end, position: " << file.tellg() << std::endl;

    // Seek backwards from end
    file.seekg(-5, std::ios::end);
    file.get(ch);
    std::cout << "Seeked to 5 chars before end, read: '" << ch << "'" << std::endl;

    // Seek relative to current position
    file.seekg(0, std::ios::beg);
    file.seekg(10, std::ios::cur);  // Move 10 positions forward from current
    file.get(ch);
    std::cout << "Seeked 10 from start (using cur), read: '" << ch << "'" << std::endl;

    file.close();

    // === 3. Reading file in reverse ===
    std::cout << "\n3. Reading file in reverse:" << std::endl;
    std::cout << "===========================" << std::endl;

    file.open("positions.txt");

    // Get file size
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    std::cout << "File size: " << fileSize << " bytes" << std::endl;

    std::cout << "Reading backwards: ";
    for (std::streampos pos = fileSize - 1; pos >= 0; pos--) {
        file.seekg(pos);
        file.get(ch);
        if (ch != '\n') {  // Skip newlines for cleaner output
            std::cout << ch;
        }
    }
    std::cout << std::endl;

    file.close();

    // === 4. tellp() and seekp() - Write position ===
    std::cout << "\n4. Write positioning (tellp/seekp):" << std::endl;
    std::cout << "===================================" << std::endl;

    // Create a file with spaces
    std::ofstream outfile("write_positions.txt");
    outfile << "____________________";  // 20 underscores
    std::cout << "Initial write position: " << outfile.tellp() << std::endl;
    outfile.close();

    // Open for reading and writing
    std::fstream rwfile("write_positions.txt", std::ios::in | std::ios::out);

    // Write at specific positions
    rwfile.seekp(0, std::ios::beg);
    rwfile << "START";
    std::cout << "After writing START, position: " << rwfile.tellp() << std::endl;

    rwfile.seekp(15, std::ios::beg);
    rwfile << "END";
    std::cout << "After writing END at pos 15, position: " << rwfile.tellp() << std::endl;

    rwfile.seekp(7, std::ios::beg);
    rwfile << "MIDDLE";

    // Read back the result
    rwfile.seekg(0, std::ios::beg);
    std::string result;
    std::getline(rwfile, result);
    std::cout << "Final content: " << result << std::endl;

    rwfile.close();

    // === 5. Practical example: Modifying specific lines ===
    std::cout << "\n5. Practical example - Modify specific line:" << std::endl;
    std::cout << "============================================" << std::endl;

    // Create config file
    std::ofstream config("config.txt");
    config << "username=admin\n";
    config << "password=old_pass\n";
    config << "port=8080\n";
    config.close();

    // Read entire file
    std::ifstream readConfig("config.txt");
    std::vector<std::string> lines;
    std::string line;

    while (std::getline(readConfig, line)) {
        lines.push_back(line);
    }
    readConfig.close();

    // Modify specific line
    for (auto& l : lines) {
        if (l.find("password=") != std::string::npos) {
            l = "password=new_secure_pass";
        }
    }

    // Write back
    std::ofstream writeConfig("config.txt");
    for (const auto& l : lines) {
        writeConfig << l << "\n";
    }
    writeConfig.close();

    std::cout << "Modified config.txt:" << std::endl;
    std::ifstream showConfig("config.txt");
    while (std::getline(showConfig, line)) {
        std::cout << "  " << line << std::endl;
    }
    showConfig.close();

    // === 6. Binary file seeking ===
    std::cout << "\n6. Binary file seeking:" << std::endl;
    std::cout << "=======================" << std::endl;

    // Write binary data
    std::ofstream binOut("numbers.bin", std::ios::binary);
    for (int i = 0; i < 10; i++) {
        binOut.write(reinterpret_cast<char*>(&i), sizeof(int));
    }
    binOut.close();

    // Read specific integers by position
    std::ifstream binIn("numbers.bin", std::ios::binary);

    // Read the 5th integer (index 4)
    binIn.seekg(4 * sizeof(int), std::ios::beg);
    int value;
    binIn.read(reinterpret_cast<char*>(&value), sizeof(int));
    std::cout << "Integer at position 4: " << value << std::endl;

    // Read the last integer
    binIn.seekg(-sizeof(int), std::ios::end);
    binIn.read(reinterpret_cast<char*>(&value), sizeof(int));
    std::cout << "Last integer: " << value << std::endl;

    binIn.close();

    std::cout << "\n✓ All seeking examples completed!" << std::endl;

    return 0;
}
