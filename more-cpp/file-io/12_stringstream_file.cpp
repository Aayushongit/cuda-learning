/**
 * 12_stringstream_file.cpp
 * Demonstrates: Combining stringstream with file I/O
 * Key Concepts:
 * - Using stringstream for in-memory string manipulation
 * - Parsing complex data from files
 * - Building formatted strings before writing
 * - Converting between types
 * - Processing structured text data
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

struct Person {
    std::string firstName;
    std::string lastName;
    int age;
    double height;
    std::string city;
};

int main() {
    std::cout << "=== StringStream with File I/O ===" << std::endl;

    // === 1. Reading and parsing with stringstream ===
    std::cout << "\n1. Parsing structured data:" << std::endl;
    std::cout << "===========================" << std::endl;

    // Create a data file
    std::ofstream create("people.txt");
    create << "John Doe 25 5.9 NewYork\n";
    create << "Jane Smith 30 5.6 LosAngeles\n";
    create << "Bob Johnson 28 6.1 Chicago\n";
    create.close();

    // Read and parse using stringstream
    std::ifstream file("people.txt");
    std::string line;
    std::vector<Person> people;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        Person p;

        ss >> p.firstName >> p.lastName >> p.age >> p.height >> p.city;
        people.push_back(p);

        std::cout << "Parsed: " << p.firstName << " " << p.lastName
                  << ", Age: " << p.age << ", Height: " << p.height
                  << ", City: " << p.city << std::endl;
    }
    file.close();

    // === 2. Building formatted output with stringstream ===
    std::cout << "\n2. Building formatted output:" << std::endl;
    std::cout << "=============================" << std::endl;

    std::ofstream report("people_report.txt");

    for (const auto& p : people) {
        std::stringstream ss;

        ss << "Name: " << std::left << std::setw(20) << (p.firstName + " " + p.lastName)
           << " | Age: " << std::setw(3) << p.age
           << " | Height: " << std::fixed << std::setprecision(1) << p.height << "ft"
           << " | City: " << p.city;

        std::string formattedLine = ss.str();
        report << formattedLine << std::endl;
        std::cout << formattedLine << std::endl;
    }
    report.close();

    // === 3. Parsing CSV with stringstream ===
    std::cout << "\n3. Parsing CSV data:" << std::endl;
    std::cout << "====================" << std::endl;

    // Create CSV file
    std::ofstream csvCreate("data.csv");
    csvCreate << "100,Widget,25.99,50\n";
    csvCreate << "101,Gadget,15.50,75\n";
    csvCreate << "102,Doohickey,99.99,10\n";
    csvCreate.close();

    std::ifstream csvFile("data.csv");

    while (std::getline(csvFile, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<std::string> fields;

        while (std::getline(ss, item, ',')) {
            fields.push_back(item);
        }

        if (fields.size() == 4) {
            int id = std::stoi(fields[0]);
            std::string name = fields[1];
            double price = std::stod(fields[2]);
            int quantity = std::stoi(fields[3]);

            std::cout << "ID: " << id << ", Product: " << name
                      << ", Price: $" << price << ", Qty: " << quantity << std::endl;
        }
    }
    csvFile.close();

    // === 4. Type conversion with stringstream ===
    std::cout << "\n4. Type conversions:" << std::endl;
    std::cout << "====================" << std::endl;

    // String to number
    std::string numStr = "12345";
    std::stringstream ss1(numStr);
    int number;
    ss1 >> number;
    std::cout << "String '" << numStr << "' to int: " << number << std::endl;

    // Number to string
    double pi = 3.14159;
    std::stringstream ss2;
    ss2 << std::fixed << std::setprecision(2) << pi;
    std::string piStr = ss2.str();
    std::cout << "Double " << pi << " to string: '" << piStr << "'" << std::endl;

    // Multiple values
    std::stringstream ss3;
    ss3 << "Value: " << 42 << ", Result: " << (3.14 * 2);
    std::cout << "Combined: " << ss3.str() << std::endl;

    // === 5. Processing complex file data ===
    std::cout << "\n5. Processing complex data:" << std::endl;
    std::cout << "===========================" << std::endl;

    // Create file with mixed format
    std::ofstream mixedFile("mixed_data.txt");
    mixedFile << "RECORD:001|NAME:Alice|SCORE:95.5|GRADE:A\n";
    mixedFile << "RECORD:002|NAME:Bob|SCORE:87.3|GRADE:B\n";
    mixedFile << "RECORD:003|NAME:Charlie|SCORE:92.8|GRADE:A\n";
    mixedFile.close();

    std::ifstream readMixed("mixed_data.txt");

    while (std::getline(readMixed, line)) {
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> parts;

        // Split by '|'
        while (std::getline(ss, segment, '|')) {
            parts.push_back(segment);
        }

        // Parse each part
        for (const auto& part : parts) {
            std::stringstream partSS(part);
            std::string key, value;

            if (std::getline(partSS, key, ':') && std::getline(partSS, value)) {
                std::cout << "  " << key << " = " << value << std::endl;
            }
        }
        std::cout << std::endl;
    }
    readMixed.close();

    // === 6. Reusing stringstream ===
    std::cout << "6. Reusing stringstream:" << std::endl;
    std::cout << "========================" << std::endl;

    std::stringstream reusableSS;

    for (int i = 1; i <= 3; i++) {
        reusableSS.str("");  // Clear content
        reusableSS.clear();  // Clear state flags

        reusableSS << "Line " << i << ": Value = " << (i * 10);
        std::cout << reusableSS.str() << std::endl;
    }

    // === 7. Reading file into stringstream ===
    std::cout << "\n7. Loading entire file into stringstream:" << std::endl;
    std::cout << "=========================================" << std::endl;

    std::ifstream input("people.txt");
    std::stringstream buffer;
    buffer << input.rdbuf();
    input.close();

    std::string fileContent = buffer.str();
    std::cout << "File content in memory:\n" << fileContent << std::endl;

    // Process the buffer
    buffer.clear();
    buffer.seekg(0);

    std::string word;
    int wordCount = 0;
    while (buffer >> word) {
        wordCount++;
    }
    std::cout << "Total words in buffer: " << wordCount << std::endl;

    // === 8. Generating structured output ===
    std::cout << "\n8. Generating JSON-like output:" << std::endl;
    std::cout << "===============================" << std::endl;

    std::ofstream jsonLike("output.json");

    for (size_t i = 0; i < people.size(); i++) {
        const auto& p = people[i];
        std::stringstream json;

        json << "{\n";
        json << "  \"id\": " << i << ",\n";
        json << "  \"firstName\": \"" << p.firstName << "\",\n";
        json << "  \"lastName\": \"" << p.lastName << "\",\n";
        json << "  \"age\": " << p.age << ",\n";
        json << "  \"height\": " << p.height << ",\n";
        json << "  \"city\": \"" << p.city << "\"\n";
        json << "}";

        if (i < people.size() - 1) {
            json << ",";
        }
        json << "\n";

        std::string jsonStr = json.str();
        jsonLike << jsonStr;
        std::cout << jsonStr;
    }
    jsonLike.close();

    // === 9. Parsing numbers from mixed content ===
    std::cout << "\n9. Extracting numbers from text:" << std::endl;
    std::cout << "================================" << std::endl;

    std::string mixedText = "The price is $45.99 and quantity is 100 units.";
    std::stringstream mixedSS(mixedText);
    std::string token;
    std::vector<double> numbers;

    while (mixedSS >> token) {
        // Try to convert to number
        std::stringstream testSS(token);
        double num;

        // Remove $ sign if present
        if (token[0] == '$') {
            token = token.substr(1);
            testSS.str(token);
        }

        if (testSS >> num) {
            numbers.push_back(num);
        }
    }

    std::cout << "Text: " << mixedText << std::endl;
    std::cout << "Extracted numbers: ";
    for (double n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    // === 10. Practical example: Config file parser ===
    std::cout << "\n10. Config file parsing:" << std::endl;
    std::cout << "========================" << std::endl;

    // Create config file
    std::ofstream configOut("app.config");
    configOut << "# Application Configuration\n";
    configOut << "host = localhost\n";
    configOut << "port = 8080\n";
    configOut << "timeout = 30\n";
    configOut << "debug = true\n";
    configOut.close();

    // Parse config
    std::ifstream configIn("app.config");
    std::cout << "Configuration settings:" << std::endl;

    while (std::getline(configIn, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        std::string key, equals, value;

        if (ss >> key >> equals >> value) {
            if (equals == "=") {
                std::cout << "  " << key << " = " << value << std::endl;
            }
        }
    }
    configIn.close();

    std::cout << "\n✓ All stringstream examples completed!" << std::endl;
    std::cout << "\nFiles created:" << std::endl;
    std::cout << "  - people.txt" << std::endl;
    std::cout << "  - people_report.txt" << std::endl;
    std::cout << "  - data.csv" << std::endl;
    std::cout << "  - mixed_data.txt" << std::endl;
    std::cout << "  - output.json" << std::endl;
    std::cout << "  - app.config" << std::endl;

    return 0;
}
