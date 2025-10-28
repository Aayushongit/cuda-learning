/**
 * 09_csv_handling.cpp
 * Demonstrates: CSV (Comma-Separated Values) file handling
 * Key Concepts:
 * - Reading CSV files
 * - Writing CSV files
 * - Parsing CSV data
 * - Handling quoted fields
 * - Working with structured data
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Employee {
    int id;
    std::string name;
    std::string department;
    double salary;
    int age;
};

// Function to split a string by delimiter
std::vector<std::string> splitCSV(const std::string& line, char delimiter = ',') {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");

        if (start != std::string::npos && end != std::string::npos) {
            tokens.push_back(token.substr(start, end - start + 1));
        } else if (start != std::string::npos) {
            tokens.push_back(token.substr(start));
        } else {
            tokens.push_back("");
        }
    }

    return tokens;
}

int main() {
    std::cout << "=== CSV File Handling ===" << std::endl;

    // === 1. Writing CSV file ===
    std::cout << "\n1. Writing CSV file:" << std::endl;
    std::cout << "====================" << std::endl;

    std::ofstream csvOut("employees.csv");

    // Write header
    csvOut << "ID,Name,Department,Salary,Age" << std::endl;

    // Write data
    csvOut << "101,Alice Johnson,Engineering,75000.50,28" << std::endl;
    csvOut << "102,Bob Smith,Marketing,65000.00,32" << std::endl;
    csvOut << "103,Charlie Davis,Engineering,80000.75,35" << std::endl;
    csvOut << "104,Diana Wilson,HR,60000.00,29" << std::endl;
    csvOut << "105,Eve Brown,Sales,70000.25,31" << std::endl;

    csvOut.close();
    std::cout << "✓ employees.csv created" << std::endl;

    // === 2. Reading CSV file ===
    std::cout << "\n2. Reading CSV file:" << std::endl;
    std::cout << "====================" << std::endl;

    std::ifstream csvIn("employees.csv");
    std::string line;

    // Read header
    std::getline(csvIn, line);
    std::cout << "Header: " << line << std::endl << std::endl;

    // Read and parse data
    std::vector<Employee> employees;

    while (std::getline(csvIn, line)) {
        std::vector<std::string> fields = splitCSV(line);

        if (fields.size() == 5) {
            Employee emp;
            emp.id = std::stoi(fields[0]);
            emp.name = fields[1];
            emp.department = fields[2];
            emp.salary = std::stod(fields[3]);
            emp.age = std::stoi(fields[4]);

            employees.push_back(emp);

            std::cout << "ID: " << emp.id << ", Name: " << emp.name
                      << ", Dept: " << emp.department << ", Salary: $" << emp.salary
                      << ", Age: " << emp.age << std::endl;
        }
    }

    csvIn.close();

    // === 3. Using stringstream for CSV parsing ===
    std::cout << "\n3. Alternative parsing with stringstream:" << std::endl;
    std::cout << "=========================================" << std::endl;

    csvIn.open("employees.csv");
    std::getline(csvIn, line);  // Skip header

    while (std::getline(csvIn, line)) {
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;

        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }

        if (fields.size() >= 2) {
            std::cout << fields[0] << " -> " << fields[1] << std::endl;
        }
    }

    csvIn.close();

    // === 4. Writing CSV with formatting ===
    std::cout << "\n4. Writing formatted CSV:" << std::endl;
    std::cout << "=========================" << std::endl;

    std::ofstream formattedCSV("sales_data.csv");

    formattedCSV << "Date,Product,Quantity,Price,Total" << std::endl;

    struct Sale {
        std::string date;
        std::string product;
        int quantity;
        double price;
    };

    Sale sales[] = {
        {"2024-01-15", "Laptop", 5, 999.99},
        {"2024-01-16", "Mouse", 25, 29.99},
        {"2024-01-17", "Keyboard", 15, 79.50},
        {"2024-01-18", "Monitor", 8, 299.00}
    };

    for (const auto& sale : sales) {
        double total = sale.quantity * sale.price;
        formattedCSV << sale.date << ","
                     << sale.product << ","
                     << sale.quantity << ","
                     << std::fixed << std::setprecision(2) << sale.price << ","
                     << total << std::endl;
    }

    formattedCSV.close();
    std::cout << "✓ sales_data.csv created" << std::endl;

    // === 5. Reading and filtering CSV data ===
    std::cout << "\n5. Filtering CSV data (Engineering dept):" << std::endl;
    std::cout << "=========================================" << std::endl;

    csvIn.open("employees.csv");
    std::getline(csvIn, line);  // Skip header

    while (std::getline(csvIn, line)) {
        std::vector<std::string> fields = splitCSV(line);

        if (fields.size() == 5 && fields[2] == "Engineering") {
            std::cout << "  " << fields[1] << " - $" << fields[3] << std::endl;
        }
    }

    csvIn.close();

    // === 6. CSV with quoted fields (handling commas in data) ===
    std::cout << "\n6. CSV with quoted fields:" << std::endl;
    std::cout << "==========================" << std::endl;

    std::ofstream quotedCSV("addresses.csv");

    quotedCSV << "ID,Name,Address,City" << std::endl;
    quotedCSV << "1,John Doe,\"123 Main St, Apt 4\",New York" << std::endl;
    quotedCSV << "2,Jane Smith,\"456 Oak Ave, Suite 200\",Los Angeles" << std::endl;
    quotedCSV << "3,Bob Johnson,789 Pine Rd,Chicago" << std::endl;

    quotedCSV.close();

    std::cout << "✓ addresses.csv created with quoted fields" << std::endl;

    // Simple reader (note: production code needs more robust quote handling)
    std::ifstream addressIn("addresses.csv");
    std::cout << "\nReading addresses:" << std::endl;

    std::getline(addressIn, line);  // Skip header

    while (std::getline(addressIn, line)) {
        std::cout << "  " << line << std::endl;
    }

    addressIn.close();

    // === 7. Data aggregation from CSV ===
    std::cout << "\n7. Data aggregation (total salary by department):" << std::endl;
    std::cout << "=================================================" << std::endl;

    csvIn.open("employees.csv");
    std::getline(csvIn, line);  // Skip header

    double engineeringTotal = 0.0, marketingTotal = 0.0, hrTotal = 0.0, salesTotal = 0.0;
    int engineeringCount = 0, marketingCount = 0, hrCount = 0, salesCount = 0;

    while (std::getline(csvIn, line)) {
        std::vector<std::string> fields = splitCSV(line);

        if (fields.size() == 5) {
            std::string dept = fields[2];
            double salary = std::stod(fields[3]);

            if (dept == "Engineering") {
                engineeringTotal += salary;
                engineeringCount++;
            } else if (dept == "Marketing") {
                marketingTotal += salary;
                marketingCount++;
            } else if (dept == "HR") {
                hrTotal += salary;
                hrCount++;
            } else if (dept == "Sales") {
                salesTotal += salary;
                salesCount++;
            }
        }
    }

    csvIn.close();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Engineering: " << engineeringCount << " employees, Total: $"
              << engineeringTotal << ", Avg: $" << (engineeringTotal / engineeringCount) << std::endl;
    std::cout << "Marketing: " << marketingCount << " employees, Total: $"
              << marketingTotal << ", Avg: $" << (marketingTotal / marketingCount) << std::endl;
    std::cout << "HR: " << hrCount << " employees, Total: $"
              << hrTotal << ", Avg: $" << (hrTotal / hrCount) << std::endl;
    std::cout << "Sales: " << salesCount << " employees, Total: $"
              << salesTotal << ", Avg: $" << (salesTotal / salesCount) << std::endl;

    // === 8. Creating summary CSV ===
    std::cout << "\n8. Creating summary report:" << std::endl;
    std::cout << "===========================" << std::endl;

    std::ofstream summaryCSV("department_summary.csv");

    summaryCSV << "Department,Employee Count,Total Salary,Average Salary" << std::endl;
    summaryCSV << std::fixed << std::setprecision(2);
    summaryCSV << "Engineering," << engineeringCount << "," << engineeringTotal << ","
               << (engineeringTotal / engineeringCount) << std::endl;
    summaryCSV << "Marketing," << marketingCount << "," << marketingTotal << ","
               << (marketingTotal / marketingCount) << std::endl;
    summaryCSV << "HR," << hrCount << "," << hrTotal << ","
               << (hrTotal / hrCount) << std::endl;
    summaryCSV << "Sales," << salesCount << "," << salesTotal << ","
               << (salesTotal / salesCount) << std::endl;

    summaryCSV.close();

    std::cout << "✓ department_summary.csv created" << std::endl;

    std::cout << "\n✓ All CSV examples completed!" << std::endl;
    std::cout << "Files created:" << std::endl;
    std::cout << "  - employees.csv" << std::endl;
    std::cout << "  - sales_data.csv" << std::endl;
    std::cout << "  - addresses.csv" << std::endl;
    std::cout << "  - department_summary.csv" << std::endl;

    return 0;
}
