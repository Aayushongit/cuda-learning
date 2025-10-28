/**
 * 04_binary_io.cpp
 * Demonstrates: Binary file I/O
 * Key Concepts:
 * - std::ios::binary mode
 * - read() and write() methods
 * - Storing structs/objects
 * - Difference between text and binary mode
 */

#include <iostream>
#include <fstream>
#include <cstring>

struct Student {
    char name[50];
    int id;
    float gpa;
    int age;
};


int main() {
    std::cout << "=== Binary File I/O ===" << std::endl;

    // === Writing Binary Data ===
    std::cout << "\n1. Writing binary data..." << std::endl;

    // Create student records
    Student students[3];

    strcpy(students[0].name, "Alice Johnson");
    students[0].id = 101;
    students[0].gpa = 3.8f;
    students[0].age = 20;

    strcpy(students[1].name, "Bob Smith");
    students[1].id = 102;
    students[1].gpa = 3.5f;
    students[1].age = 21;

    strcpy(students[2].name, "Charlie Davis");
    students[2].id = 103;
    students[2].gpa = 3.9f;
    students[2].age = 19;

    // Write to binary file
    std::ofstream binOut("students.dat", std::ios::binary);

    if (!binOut) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return 1;
    }

    // Write entire array
    binOut.write(reinterpret_cast<char*>(students), sizeof(students));
    binOut.close();

    std::cout << "Written " << sizeof(students) << " bytes to students.dat" << std::endl;

    // === Reading Binary Data ===
    std::cout << "\n2. Reading binary data..." << std::endl;

    Student readStudents[3];
    std::ifstream binIn("students.dat", std::ios::binary);

    if (!binIn) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return 1;
    }

    binIn.read(reinterpret_cast<char*>(readStudents), sizeof(readStudents));
    binIn.close();

    std::cout << "\nStudent Records:" << std::endl;
    std::cout << "=================" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "Student " << (i + 1) << ":" << std::endl;
        std::cout << "  Name: " << readStudents[i].name << std::endl;
        std::cout << "  ID: " << readStudents[i].id << std::endl;
        std::cout << "  GPA: " << readStudents[i].gpa << std::endl;
        std::cout << "  Age: " << readStudents[i].age << std::endl;
        std::cout << std::endl;
    }

    // === Comparing Text vs Binary ===
    std::cout << "3. Comparing text vs binary storage..." << std::endl;

    // Write numbers as text
    std::ofstream textFile("numbers_text.txt");
    for (int i = 0; i < 100; i++) {
        textFile << i << " ";
    }
    textFile.close();

    // Write numbers as binary
    std::ofstream binaryFile("numbers_binary.dat", std::ios::binary);
    for (int i = 0; i < 100; i++) {
        binaryFile.write(reinterpret_cast<char*>(&i), sizeof(int));
    }
    binaryFile.close();

    // Check file sizes
    std::ifstream textCheck("numbers_text.txt", std::ios::ate);
    std::ifstream binaryCheck("numbers_binary.dat", std::ios::ate);

    std::cout << "Text file size: " << textCheck.tellg() << " bytes" << std::endl;
    std::cout << "Binary file size: " << binaryCheck.tellg() << " bytes" << std::endl;

    textCheck.close();
    binaryCheck.close();

    // === Writing individual binary values ===
    std::cout << "\n4. Writing individual binary values..." << std::endl;

    std::ofstream mixedBinary("mixed_data.bin", std::ios::binary);

    int intVal = 12345;
    double doubleVal = 3.14159265359;
    char charVal = 'X';
    bool boolVal = true;

    mixedBinary.write(reinterpret_cast<char*>(&intVal), sizeof(intVal));
    mixedBinary.write(reinterpret_cast<char*>(&doubleVal), sizeof(doubleVal));
    mixedBinary.write(&charVal, sizeof(charVal));
    mixedBinary.write(reinterpret_cast<char*>(&boolVal), sizeof(boolVal));

    mixedBinary.close();

    // Read them back
    std::ifstream mixedRead("mixed_data.bin", std::ios::binary);

    int readInt;
    double readDouble;
    char readChar;
    bool readBool;

    mixedRead.read(reinterpret_cast<char*>(&readInt), sizeof(readInt));
    mixedRead.read(reinterpret_cast<char*>(&readDouble), sizeof(readDouble));
    mixedRead.read(&readChar, sizeof(readChar));
    mixedRead.read(reinterpret_cast<char*>(&readBool), sizeof(readBool));

    mixedRead.close();

    std::cout << "Read values:" << std::endl;
    std::cout << "  Int: " << readInt << std::endl;
    std::cout << "  Double: " << readDouble << std::endl;
    std::cout << "  Char: " << readChar << std::endl;
    std::cout << "  Bool: " << std::boolalpha << readBool << std::endl;

    return 0;
}
