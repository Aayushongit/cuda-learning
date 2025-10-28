/**
 * 13_streams_io.cpp
 *
 * STREAMS AND I/O
 * - cout, cin, cerr (console I/O)
 * - ifstream, ofstream, fstream (file I/O)
 * - stringstream (string I/O)
 * - I/O manipulators
 * - Binary I/O
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== STREAMS AND I/O ===\n";

    separator("CONSOLE OUTPUT");

    // 1. Basic Output
    std::cout << "\n1. BASIC OUTPUT:\n";
    std::cout << "Hello, World!\n";
    std::cout << "Number: " << 42 << "\n";
    std::cout << "Multiple " << "values " << "in " << "one " << "statement\n";

    // 2. Output Manipulators
    std::cout << "\n2. OUTPUT MANIPULATORS:\n";
    std::cout << "Default: " << 123.456789 << "\n";
    std::cout << std::fixed << std::setprecision(2) << "Fixed 2 decimals: " << 123.456789 << "\n";
    std::cout << std::scientific << "Scientific: " << 123.456789 << "\n";
    std::cout << std::defaultfloat;  // Reset

    // Width and fill
    std::cout << std::setw(10) << std::setfill('*') << 42 << "\n";
    std::cout << std::setw(10) << std::setfill(' ') << std::left << "Left" << "|\n";
    std::cout << std::setw(10) << std::right << "Right" << "|\n";

    // Boolean
    std::cout << std::boolalpha << "true is " << true << "\n";
    std::cout << std::noboolalpha << "true is " << true << "\n";

    // Hex, Oct, Dec
    int num = 255;
    std::cout << "Dec: " << std::dec << num << "\n";
    std::cout << "Hex: " << std::hex << num << "\n";
    std::cout << "Oct: " << std::oct << num << "\n";
    std::cout << std::dec;  // Reset to decimal

    // 3. cerr and clog
    std::cout << "\n3. CERR AND CLOG:\n";
    std::cerr << "This is an error message (unbuffered)\n";
    std::clog << "This is a log message (buffered)\n";

    separator("FILE OUTPUT");

    // 4. Writing to File
    std::cout << "\n4. WRITING TO FILE:\n";
    {
        std::ofstream outfile("test_output.txt");
        if (outfile.is_open()) {
            outfile << "Line 1\n";
            outfile << "Line 2\n";
            outfile << "Number: " << 42 << "\n";
            outfile.close();
            std::cout << "File written successfully\n";
        } else {
            std::cerr << "Failed to open file for writing\n";
        }
    }

    // 5. Append Mode
    std::cout << "\n5. APPEND MODE:\n";
    {
        std::ofstream appendfile("test_output.txt", std::ios::app);
        if (appendfile.is_open()) {
            appendfile << "Appended line\n";
            appendfile.close();
            std::cout << "Data appended\n";
        }
    }

    separator("FILE INPUT");

    // 6. Reading from File
    std::cout << "\n6. READING FROM FILE:\n";
    {
        std::ifstream infile("test_output.txt");
        if (infile.is_open()) {
            std::string line;
            while (std::getline(infile, line)) {
                std::cout << "Read: " << line << "\n";
            }
            infile.close();
        } else {
            std::cerr << "Failed to open file for reading\n";
        }
    }

    // 7. Read Word by Word
    std::cout << "\n7. READ WORD BY WORD:\n";
    {
        std::ifstream infile("test_output.txt");
        if (infile.is_open()) {
            std::string word;
            std::cout << "Words: ";
            while (infile >> word) {
                std::cout << word << " ";
            }
            std::cout << "\n";
            infile.close();
        }
    }

    // 8. Check File State
    std::cout << "\n8. FILE STATE:\n";
    {
        std::ifstream file("test_output.txt");
        if (file.good()) std::cout << "File is good\n";
        if (file.eof()) std::cout << "Reached EOF\n";
        if (file.fail()) std::cout << "Operation failed\n";
        if (file.bad()) std::cout << "Read/write error\n";

        // Read entire file
        std::string content;
        while (file >> content);

        if (file.eof()) std::cout << "Now at EOF\n";
        file.close();
    }

    separator("FSTREAM (READ AND WRITE)");

    // 9. Read and Write
    std::cout << "\n9. FSTREAM:\n";
    {
        std::fstream file("readwrite.txt", std::ios::out | std::ios::in | std::ios::trunc);
        if (file.is_open()) {
            // Write
            file << "Data written via fstream\n";

            // Move to beginning
            file.seekg(0, std::ios::beg);

            // Read
            std::string line;
            std::getline(file, line);
            std::cout << "Read back: " << line << "\n";

            file.close();
        }
    }

    separator("STRING STREAMS");

    // 10. ostringstream
    std::cout << "\n10. OSTRINGSTREAM:\n";
    std::ostringstream oss;
    oss << "Number: " << 42 << ", Pi: " << 3.14159;
    std::string result = oss.str();
    std::cout << "String stream result: " << result << "\n";

    // 11. istringstream
    std::cout << "\n11. ISTRINGSTREAM:\n";
    std::istringstream iss("10 20 30");
    int a, b, c;
    iss >> a >> b >> c;
    std::cout << "Parsed: a=" << a << ", b=" << b << ", c=" << c << "\n";

    // 12. stringstream
    std::cout << "\n12. STRINGSTREAM:\n";
    std::stringstream ss;
    ss << "Value: " << 100;
    std::string str = ss.str();
    std::cout << "Output: " << str << "\n";

    ss.str("");  // Clear
    ss.clear();  // Reset state

    ss << "200 300";
    int x, y;
    ss >> x >> y;
    std::cout << "Parsed: x=" << x << ", y=" << y << "\n";

    separator("BINARY I/O");

    // 13. Write Binary
    std::cout << "\n13. BINARY WRITE:\n";
    {
        std::ofstream binfile("binary.dat", std::ios::binary);
        if (binfile.is_open()) {
            int numbers[] = {10, 20, 30, 40, 50};
            binfile.write(reinterpret_cast<char*>(numbers), sizeof(numbers));
            binfile.close();
            std::cout << "Binary data written\n";
        }
    }

    // 14. Read Binary
    std::cout << "\n14. BINARY READ:\n";
    {
        std::ifstream binfile("binary.dat", std::ios::binary);
        if (binfile.is_open()) {
            int numbers[5];
            binfile.read(reinterpret_cast<char*>(numbers), sizeof(numbers));
            std::cout << "Read binary: ";
            for (int n : numbers) {
                std::cout << n << " ";
            }
            std::cout << "\n";
            binfile.close();
        }
    }

    separator("FILE POSITIONING");

    // 15. tellg and seekg
    std::cout << "\n15. FILE POSITIONING:\n";
    {
        std::ofstream out("position.txt");
        out << "0123456789ABCDEF";
        out.close();

        std::ifstream in("position.txt");
        if (in.is_open()) {
            std::cout << "Position: " << in.tellg() << "\n";

            char ch;
            in >> ch;
            std::cout << "Read: " << ch << ", Position: " << in.tellg() << "\n";

            // Seek to position 5
            in.seekg(5, std::ios::beg);
            in >> ch;
            std::cout << "After seek(5): " << ch << "\n";

            // Seek from current
            in.seekg(2, std::ios::cur);
            in >> ch;
            std::cout << "After seek(2, cur): " << ch << "\n";

            // Seek from end
            in.seekg(-3, std::ios::end);
            in >> ch;
            std::cout << "After seek(-3, end): " << ch << "\n";

            in.close();
        }
    }

    separator("FORMATTING");

    // 16. Table Formatting
    std::cout << "\n16. TABLE FORMATTING:\n";
    std::cout << std::left << std::setw(15) << "Name"
              << std::right << std::setw(10) << "Age"
              << std::right << std::setw(12) << "Salary\n";
    std::cout << std::string(37, '-') << "\n";

    std::cout << std::left << std::setw(15) << "Alice"
              << std::right << std::setw(10) << 30
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << 50000.00 << "\n";

    std::cout << std::left << std::setw(15) << "Bob"
              << std::right << std::setw(10) << 25
              << std::right << std::setw(12) << 45000.50 << "\n";

    std::cout << "\n=== END OF STREAMS AND I/O ===\n";

    return 0;
}
