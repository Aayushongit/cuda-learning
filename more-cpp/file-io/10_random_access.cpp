/**
 * 10_random_access.cpp
 * Demonstrates: Random access file operations
 * Key Concepts:
 * - Random access to records
 * - Fixed-size records
 * - Direct positioning
 * - Reading/writing at specific positions
 * - Building a simple database
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>

struct Record {
    int id;
    char name[50];
    double balance;
    bool active;

    Record() : id(0), balance(0.0), active(false) {
        memset(name, 0, sizeof(name));
    }

    Record(int i, const std::string& n, double b, bool a)
        : id(i), balance(b), active(a) {
        strncpy(name, n.c_str(), sizeof(name) - 1);
        name[sizeof(name) - 1] = '\0';
    }
};

class RecordDatabase {
private:
    std::string filename;

public:
    RecordDatabase(const std::string& fname) : filename(fname) {}

    // Write a record at specific position
    bool writeRecord(int position, const Record& record) {
        std::fstream file(filename, std::ios::in | std::ios::out | std::ios::binary);

        if (!file) {
            // File doesn't exist, create it
            file.open(filename, std::ios::out | std::ios::binary);
            file.close();
            file.open(filename, std::ios::in | std::ios::out | std::ios::binary);
        }

        if (!file) {
            return false;
        }

        // Seek to position
        file.seekp(position * sizeof(Record), std::ios::beg);

        // Write record
        file.write(reinterpret_cast<const char*>(&record), sizeof(Record));

        file.close();
        return true;
    }

    // Read a record from specific position
    bool readRecord(int position, Record& record) {
        std::ifstream file(filename, std::ios::binary);

        if (!file) {
            return false;
        }

        // Seek to position
        file.seekg(position * sizeof(Record), std::ios::beg);

        // Check if position is valid
        if (file.tellg() == -1) {
            file.close();
            return false;
        }

        // Read record
        file.read(reinterpret_cast<char*>(&record), sizeof(Record));

        bool success = file.good() || file.eof();
        file.close();
        return success;
    }

    // Update a record
    bool updateRecord(int position, const Record& record) {
        return writeRecord(position, record);
    }

    // Get total number of records
    int getRecordCount() {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);

        if (!file) {
            return 0;
        }

        std::streampos fileSize = file.tellg();
        file.close();

        return static_cast<int>(fileSize / sizeof(Record));
    }

    // Display all records
    void displayAll() {
        int count = getRecordCount();

        std::cout << "\n=== All Records ===" << std::endl;
        std::cout << "Total records: " << count << std::endl << std::endl;

        for (int i = 0; i < count; i++) {
            Record record;
            if (readRecord(i, record)) {
                std::cout << "Position " << i << ":" << std::endl;
                std::cout << "  ID: " << record.id << std::endl;
                std::cout << "  Name: " << record.name << std::endl;
                std::cout << "  Balance: $" << std::fixed << std::setprecision(2)
                          << record.balance << std::endl;
                std::cout << "  Active: " << (record.active ? "Yes" : "No") << std::endl;
                std::cout << std::endl;
            }
        }
    }
};

int main() {
    std::cout << "=== Random Access File Operations ===" << std::endl;

    // === 1. Creating a random access database ===
    std::cout << "\n1. Creating random access database:" << std::endl;
    std::cout << "===================================" << std::endl;

    RecordDatabase db("accounts.dat");

    // Create some records
    Record r1(1001, "Alice Johnson", 5000.50, true);
    Record r2(1002, "Bob Smith", 3200.75, true);
    Record r3(1003, "Charlie Davis", 7500.00, true);
    Record r4(1004, "Diana Wilson", 2100.25, false);
    Record r5(1005, "Eve Brown", 9800.00, true);

    // Write records at specific positions
    db.writeRecord(0, r1);
    db.writeRecord(1, r2);
    db.writeRecord(2, r3);
    db.writeRecord(3, r4);
    db.writeRecord(4, r5);

    std::cout << "✓ 5 records written to database" << std::endl;

    // === 2. Random access reading ===
    std::cout << "\n2. Reading specific records:" << std::endl;
    std::cout << "============================" << std::endl;

    // Read record at position 2
    Record readRecord;
    if (db.readRecord(2, readRecord)) {
        std::cout << "Record at position 2:" << std::endl;
        std::cout << "  ID: " << readRecord.id << std::endl;
        std::cout << "  Name: " << readRecord.name << std::endl;
        std::cout << "  Balance: $" << std::fixed << std::setprecision(2)
                  << readRecord.balance << std::endl;
    }

    // Read record at position 4
    if (db.readRecord(4, readRecord)) {
        std::cout << "\nRecord at position 4:" << std::endl;
        std::cout << "  ID: " << readRecord.id << std::endl;
        std::cout << "  Name: " << readRecord.name << std::endl;
        std::cout << "  Balance: $" << readRecord.balance << std::endl;
    }

    // === 3. Updating a record ===
    std::cout << "\n3. Updating a record:" << std::endl;
    std::cout << "=====================" << std::endl;

    // Read current record
    db.readRecord(1, readRecord);
    std::cout << "Before update:" << std::endl;
    std::cout << "  Name: " << readRecord.name << std::endl;
    std::cout << "  Balance: $" << readRecord.balance << std::endl;

    // Modify and update
    readRecord.balance += 1000.00;  // Add $1000
    db.updateRecord(1, readRecord);

    // Verify update
    Record updatedRecord;
    db.readRecord(1, updatedRecord);
    std::cout << "\nAfter update:" << std::endl;
    std::cout << "  Name: " << updatedRecord.name << std::endl;
    std::cout << "  Balance: $" << updatedRecord.balance << std::endl;

    // === 4. Display all records ===
    db.displayAll();

    // === 5. Direct binary file random access ===
    std::cout << "\n5. Direct binary random access:" << std::endl;
    std::cout << "===============================" << std::endl;

    // Create a file with integers
    std::ofstream binOut("numbers.dat", std::ios::binary);
    for (int i = 0; i < 20; i++) {
        int value = i * 10;
        binOut.write(reinterpret_cast<char*>(&value), sizeof(int));
    }
    binOut.close();

    std::cout << "Written 20 integers (0, 10, 20, ..., 190)" << std::endl;

    // Random access read
    std::fstream binFile("numbers.dat", std::ios::in | std::ios::binary);

    // Read integers at specific positions
    int positions[] = {5, 10, 15, 0, 19};

    std::cout << "\nReading integers at random positions:" << std::endl;
    for (int pos : positions) {
        binFile.seekg(pos * sizeof(int), std::ios::beg);

        int value;
        binFile.read(reinterpret_cast<char*>(&value), sizeof(int));

        std::cout << "  Position " << pos << ": " << value << std::endl;
    }

    binFile.close();

    // === 6. Modifying binary file in place ===
    std::cout << "\n6. Modifying binary file in place:" << std::endl;
    std::cout << "===================================" << std::endl;

    // Open for both reading and writing
    std::fstream binRW("numbers.dat", std::ios::in | std::ios::out | std::ios::binary);

    // Double the value at position 10
    binRW.seekg(10 * sizeof(int), std::ios::beg);
    int value;
    binRW.read(reinterpret_cast<char*>(&value), sizeof(int));

    std::cout << "Value at position 10 before: " << value << std::endl;

    value *= 2;

    binRW.seekp(10 * sizeof(int), std::ios::beg);
    binRW.write(reinterpret_cast<char*>(&value), sizeof(int));

    // Verify
    binRW.seekg(10 * sizeof(int), std::ios::beg);
    binRW.read(reinterpret_cast<char*>(&value), sizeof(int));

    std::cout << "Value at position 10 after: " << value << std::endl;

    binRW.close();

    // === 7. Sparse file (writing at non-consecutive positions) ===
    std::cout << "\n7. Creating sparse file:" << std::endl;
    std::cout << "========================" << std::endl;

    RecordDatabase sparseDB("sparse.dat");

    // Write records at non-consecutive positions
    Record sparse1(9001, "Record at 0", 100.0, true);
    Record sparse2(9005, "Record at 5", 500.0, true);
    Record sparse3(9010, "Record at 10", 1000.0, true);

    sparseDB.writeRecord(0, sparse1);
    sparseDB.writeRecord(5, sparse2);
    sparseDB.writeRecord(10, sparse3);

    std::cout << "✓ Created sparse file with records at positions 0, 5, 10" << std::endl;
    std::cout << "Total file size: " << (sparseDB.getRecordCount() * sizeof(Record))
              << " bytes" << std::endl;

    std::cout << "\n✓ All random access examples completed!" << std::endl;

    return 0;
}
