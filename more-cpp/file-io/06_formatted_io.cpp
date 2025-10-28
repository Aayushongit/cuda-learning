/**
 * 06_formatted_io.cpp
 * Demonstrates: Formatted input/output
 * Key Concepts:
 * - I/O manipulators (setw, setprecision, setfill)
 * - Field width and alignment
 * - Number formatting
 * - Creating tables and reports
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

struct Product {
    int id;
    std::string name;
    double price;
    int quantity;
};

int main() {
    std::cout << "=== Formatted File I/O ===" << std::endl;

    // === 1. Basic formatting with manipulators ===
    std::ofstream report("sales_report.txt");

    // Write header
    report << std::left << std::setw(10) << "ID"
           << std::setw(20) << "Product"
           << std::right << std::setw(12) << "Price"
           << std::setw(12) << "Quantity"
           << std::setw(12) << "Total" << std::endl;

    report << std::string(66, '-') << std::endl;

    // Product data
    Product products[] = {
        {101, "Laptop", 999.99, 5},
        {102, "Mouse", 29.99, 15},
        {103, "Keyboard", 79.50, 8},
        {104, "Monitor", 299.00, 3},
        {105, "USB Cable", 12.99, 25}
    };

    // Write formatted data
    double grandTotal = 0.0;

    for (const auto& p : products) {
        double total = p.price * p.quantity;
        grandTotal += total;

        report << std::left << std::setw(10) << p.id
               << std::setw(20) << p.name
               << std::right << std::fixed << std::setprecision(2)
               << std::setw(12) << p.price
               << std::setw(12) << p.quantity
               << std::setw(12) << total << std::endl;
    }

    report << std::string(66, '-') << std::endl;
    report << std::right << std::setw(54) << "Grand Total: "
           << std::setw(12) << grandTotal << std::endl;

    report.close();
    std::cout << "Sales report created!" << std::endl;

    // === 2. Number formatting examples ===
    std::ofstream numbers("number_formats.txt");

    double pi = 3.14159265359;
    int year = 2024;

    numbers << "=== Number Formatting Examples ===" << std::endl << std::endl;

    numbers << "Default: " << pi << std::endl;
    numbers << "Fixed 2 decimals: " << std::fixed << std::setprecision(2) << pi << std::endl;
    numbers << "Fixed 5 decimals: " << std::fixed << std::setprecision(5) << pi << std::endl;
    numbers << "Scientific: " << std::scientific << pi << std::endl;
    numbers << "Default (reset): " << std::defaultfloat << pi << std::endl;

    numbers << std::endl;

    numbers << "Integer width 10, fill with zeros: "
            << std::setfill('0') << std::setw(10) << year << std::endl;
    numbers << "Integer width 10, fill with spaces: "
            << std::setfill(' ') << std::setw(10) << year << std::endl;

    numbers << std::endl;

    // Alignment examples
    numbers << "Left aligned: |" << std::left << std::setw(20) << "Hello" << "|" << std::endl;
    numbers << "Right aligned: |" << std::right << std::setw(20) << "Hello" << "|" << std::endl;

    numbers.close();

    // === 3. Boolean formatting ===
    std::ofstream bools("bool_formats.txt");

    bool flag1 = true;
    bool flag2 = false;

    bools << "Default bool format:" << std::endl;
    bools << "flag1: " << flag1 << ", flag2: " << flag2 << std::endl;

    bools << std::endl << "Alpha bool format:" << std::endl;
    bools << std::boolalpha;
    bools << "flag1: " << flag1 << ", flag2: " << flag2 << std::endl;

    bools.close();

    // === 4. Hex, Oct, Dec formatting ===
    std::ofstream bases("number_bases.txt");

    int num = 255;

    bases << "Number: " << num << std::endl;
    bases << "Decimal: " << std::dec << num << std::endl;
    bases << "Hexadecimal: " << std::hex << num << std::endl;
    bases << "Octal: " << std::oct << num << std::endl;

    bases << std::endl << "With prefixes:" << std::endl;
    bases << std::showbase;
    bases << "Decimal: " << std::dec << num << std::endl;
    bases << "Hexadecimal: " << std::hex << num << std::endl;
    bases << "Octal: " << std::oct << num << std::endl;

    bases.close();

    // === 5. Creating a fancy bordered table ===
    std::ofstream table("fancy_table.txt");

    table << "╔════════════════════════════════════════════════════╗" << std::endl;
    table << "║            SYSTEM STATUS REPORT                    ║" << std::endl;
    table << "╠════════════════════════════════════════════════════╣" << std::endl;

    table << "║ " << std::left << std::setw(25) << "CPU Usage"
          << std::right << std::setw(20) << "45.2%" << "   ║" << std::endl;

    table << "║ " << std::left << std::setw(25) << "Memory Usage"
          << std::right << std::setw(20) << "2.4 GB / 8.0 GB" << "   ║" << std::endl;

    table << "║ " << std::left << std::setw(25) << "Disk Space"
          << std::right << std::setw(20) << "156 GB / 500 GB" << "   ║" << std::endl;

    table << "║ " << std::left << std::setw(25) << "Uptime"
          << std::right << std::setw(20) << "5 days, 3 hours" << "   ║" << std::endl;

    table << "╚════════════════════════════════════════════════════╝" << std::endl;

    table.close();

    // === Display all created files ===
    std::cout << "\n--- Sales Report ---" << std::endl;
    std::ifstream showReport("sales_report.txt");
    std::string line;
    while (std::getline(showReport, line)) {
        std::cout << line << std::endl;
    }
    showReport.close();

    std::cout << "\n--- Fancy Table ---" << std::endl;
    std::ifstream showTable("fancy_table.txt");
    while (std::getline(showTable, line)) {
        std::cout << line << std::endl;
    }
    showTable.close();

    std::cout << "\n✓ All formatted files created successfully!" << std::endl;
    std::cout << "  - sales_report.txt" << std::endl;
    std::cout << "  - number_formats.txt" << std::endl;
    std::cout << "  - bool_formats.txt" << std::endl;
    std::cout << "  - number_bases.txt" << std::endl;
    std::cout << "  - fancy_table.txt" << std::endl;

    return 0;
}
