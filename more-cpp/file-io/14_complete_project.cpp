/**
 * 14_complete_project.cpp
 * Complete Project: Student Grade Management System
 * Combines all file I/O concepts:
 * - Reading/Writing text and binary files
 * - CSV handling
 * - Random access
 * - Error handling
 * - File utilities
 * - Formatted output
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cstring>
#include <algorithm>

// Student structure
struct Student {
    int id;
    char name[50];
    char course[30];
    float grade;
    bool active;

    Student() : id(0), grade(0.0f), active(true) {
        memset(name, 0, sizeof(name));
        memset(course, 0, sizeof(course));
    }

    Student(int i, const std::string& n, const std::string& c, float g)
        : id(i), grade(g), active(true) {
        strncpy(name, n.c_str(), sizeof(name) - 1);
        strncpy(course, c.c_str(), sizeof(course) - 1);
    }
};

// Main application class
class GradeManagementSystem {
private:
    std::string dataFile;
    std::vector<Student> students;

    void logOperation(const std::string& operation) {
        std::ofstream logFile("system.log", std::ios::app);
        if (logFile.is_open()) {
            time_t now = time(nullptr);
            char timeStr[26];
            ctime_r(&now, timeStr);
            timeStr[24] = '\0';  // Remove newline
            logFile << "[" << timeStr << "] " << operation << std::endl;
            logFile.close();
        }
    }

public:
    GradeManagementSystem(const std::string& file) : dataFile(file) {
        logOperation("System initialized");
    }

    // Load students from binary file
    bool loadFromBinary() {
        std::ifstream file(dataFile, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "No existing data file found. Starting fresh." << std::endl;
            return false;
        }

        students.clear();
        Student temp;

        while (file.read(reinterpret_cast<char*>(&temp), sizeof(Student))) {
            students.push_back(temp);
        }

        file.close();
        logOperation("Loaded " + std::to_string(students.size()) + " students from binary file");
        std::cout << "✓ Loaded " << students.size() << " students" << std::endl;
        return true;
    }

    // Save students to binary file
    bool saveToBinary() {
        std::ofstream file(dataFile, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "✗ Error: Cannot save to file" << std::endl;
            return false;
        }

        for (const auto& student : students) {
            file.write(reinterpret_cast<const char*>(&student), sizeof(Student));
        }

        file.close();
        logOperation("Saved " + std::to_string(students.size()) + " students to binary file");
        std::cout << "✓ Data saved successfully" << std::endl;
        return true;
    }

    // Add new student
    void addStudent(int id, const std::string& name, const std::string& course, float grade) {
        Student s(id, name, course, grade);
        students.push_back(s);
        logOperation("Added student: " + std::string(s.name) + " (ID: " + std::to_string(id) + ")");
        std::cout << "✓ Student added: " << name << std::endl;
    }

    // Find student by ID
    Student* findStudent(int id) {
        for (auto& s : students) {
            if (s.id == id && s.active) {
                return &s;
            }
        }
        return nullptr;
    }

    // Update student grade
    bool updateGrade(int id, float newGrade) {
        Student* s = findStudent(id);
        if (s) {
            s->grade = newGrade;
            logOperation("Updated grade for student ID " + std::to_string(id));
            std::cout << "✓ Grade updated for " << s->name << std::endl;
            return true;
        }
        std::cout << "✗ Student not found" << std::endl;
        return false;
    }

    // Delete student (soft delete)
    bool deleteStudent(int id) {
        Student* s = findStudent(id);
        if (s) {
            s->active = false;
            logOperation("Deleted student: " + std::string(s->name) + " (ID: " + std::to_string(id) + ")");
            std::cout << "✓ Student deleted" << std::endl;
            return true;
        }
        std::cout << "✗ Student not found" << std::endl;
        return false;
    }

    // Display all active students
    void displayAllStudents() {
        std::cout << "\n=== All Active Students ===" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << std::left << std::setw(8) << "ID"
                  << std::setw(25) << "Name"
                  << std::setw(20) << "Course"
                  << std::right << std::setw(10) << "Grade" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        int count = 0;
        for (const auto& s : students) {
            if (s.active) {
                std::cout << std::left << std::setw(8) << s.id
                          << std::setw(25) << s.name
                          << std::setw(20) << s.course
                          << std::right << std::fixed << std::setprecision(2)
                          << std::setw(10) << s.grade << std::endl;
                count++;
            }
        }

        std::cout << std::string(70, '-') << std::endl;
        std::cout << "Total active students: " << count << std::endl;
    }

    // Export to CSV
    bool exportToCSV(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "✗ Error: Cannot create CSV file" << std::endl;
            return false;
        }

        // Write header
        file << "ID,Name,Course,Grade,Status" << std::endl;

        // Write data
        for (const auto& s : students) {
            file << s.id << ","
                 << s.name << ","
                 << s.course << ","
                 << std::fixed << std::setprecision(2) << s.grade << ","
                 << (s.active ? "Active" : "Inactive") << std::endl;
        }

        file.close();
        logOperation("Exported to CSV: " + filename);
        std::cout << "✓ Exported to " << filename << std::endl;
        return true;
    }

    // Import from CSV
    bool importFromCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "✗ Error: Cannot open CSV file" << std::endl;
            return false;
        }

        std::string line;
        std::getline(file, line);  // Skip header

        int imported = 0;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string field;
            std::vector<std::string> fields;

            while (std::getline(ss, field, ',')) {
                fields.push_back(field);
            }

            if (fields.size() >= 4) {
                try {
                    int id = std::stoi(fields[0]);
                    std::string name = fields[1];
                    std::string course = fields[2];
                    float grade = std::stof(fields[3]);

                    addStudent(id, name, course, grade);
                    imported++;
                } catch (...) {
                    std::cerr << "Warning: Skipped invalid line" << std::endl;
                }
            }
        }

        file.close();
        logOperation("Imported " + std::to_string(imported) + " students from CSV");
        std::cout << "✓ Imported " << imported << " students" << std::endl;
        return true;
    }

    // Generate report
    void generateReport(const std::string& filename) {
        std::ofstream report(filename);
        if (!report.is_open()) {
            std::cerr << "✗ Error: Cannot create report" << std::endl;
            return;
        }

        // Calculate statistics
        float totalGrade = 0.0f;
        int activeCount = 0;
        float highest = 0.0f;
        float lowest = 100.0f;

        for (const auto& s : students) {
            if (s.active) {
                totalGrade += s.grade;
                activeCount++;
                highest = std::max(highest, s.grade);
                lowest = std::min(lowest, s.grade);
            }
        }

        float average = activeCount > 0 ? totalGrade / activeCount : 0.0f;

        // Write report
        report << "╔════════════════════════════════════════════════════╗" << std::endl;
        report << "║        STUDENT GRADE MANAGEMENT REPORT             ║" << std::endl;
        report << "╚════════════════════════════════════════════════════╝" << std::endl;
        report << std::endl;

        report << "Total Students: " << students.size() << std::endl;
        report << "Active Students: " << activeCount << std::endl;
        report << "Inactive Students: " << (students.size() - activeCount) << std::endl;
        report << std::endl;

        report << "Grade Statistics:" << std::endl;
        report << std::fixed << std::setprecision(2);
        report << "  Average Grade: " << average << std::endl;
        report << "  Highest Grade: " << highest << std::endl;
        report << "  Lowest Grade:  " << lowest << std::endl;
        report << std::endl;

        report << std::string(70, '=') << std::endl;
        report << std::left << std::setw(8) << "ID"
               << std::setw(25) << "Name"
               << std::setw(20) << "Course"
               << std::right << std::setw(10) << "Grade" << std::endl;
        report << std::string(70, '=') << std::endl;

        for (const auto& s : students) {
            if (s.active) {
                report << std::left << std::setw(8) << s.id
                       << std::setw(25) << s.name
                       << std::setw(20) << s.course
                       << std::right << std::fixed << std::setprecision(2)
                       << std::setw(10) << s.grade << std::endl;
            }
        }

        report.close();
        logOperation("Generated report: " + filename);
        std::cout << "✓ Report generated: " << filename << std::endl;
    }

    // Backup system
    bool createBackup() {
        std::string backupFile = dataFile + ".backup";
        std::ifstream src(dataFile, std::ios::binary);
        std::ofstream dst(backupFile, std::ios::binary);

        if (!src || !dst) {
            std::cerr << "✗ Error: Cannot create backup" << std::endl;
            return false;
        }

        dst << src.rdbuf();
        src.close();
        dst.close();

        logOperation("Backup created: " + backupFile);
        std::cout << "✓ Backup created" << std::endl;
        return true;
    }
};

// Demo menu system
void displayMenu() {
    std::cout << "\n╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   STUDENT GRADE MANAGEMENT SYSTEM              ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl;
    std::cout << "1. Add Student" << std::endl;
    std::cout << "2. Update Grade" << std::endl;
    std::cout << "3. Delete Student" << std::endl;
    std::cout << "4. Display All Students" << std::endl;
    std::cout << "5. Export to CSV" << std::endl;
    std::cout << "6. Import from CSV" << std::endl;
    std::cout << "7. Generate Report" << std::endl;
    std::cout << "8. Create Backup" << std::endl;
    std::cout << "9. Save & Exit" << std::endl;
    std::cout << "Choice: ";
}

int main() {
    std::cout << "=== Complete File I/O Project ===" << std::endl;
    std::cout << "Student Grade Management System" << std::endl;
    std::cout << std::endl;

    GradeManagementSystem system("students.dat");

    // Try to load existing data
    system.loadFromBinary();

    // Demo: Add sample students
    std::cout << "\nAdding sample students..." << std::endl;
    system.addStudent(1001, "Alice Johnson", "Computer Science", 92.5f);
    system.addStudent(1002, "Bob Smith", "Mathematics", 87.0f);
    system.addStudent(1003, "Charlie Davis", "Physics", 95.5f);
    system.addStudent(1004, "Diana Wilson", "Chemistry", 88.5f);
    system.addStudent(1005, "Eve Brown", "Biology", 91.0f);

    // Display all students
    system.displayAllStudents();

    // Update a grade
    std::cout << "\nUpdating grade for student 1002..." << std::endl;
    system.updateGrade(1002, 89.5f);

    // Delete a student
    std::cout << "\nDeleting student 1004..." << std::endl;
    system.deleteStudent(1004);

    // Display updated list
    system.displayAllStudents();

    // Export to CSV
    system.exportToCSV("students.csv");

    // Generate report
    system.generateReport("grade_report.txt");

    // Create backup
    system.createBackup();

    // Save to binary file
    system.saveToBinary();

    // Display the generated report
    std::cout << "\n=== Generated Report ===" << std::endl;
    std::ifstream reportFile("grade_report.txt");
    std::string line;
    while (std::getline(reportFile, line)) {
        std::cout << line << std::endl;
    }
    reportFile.close();

    std::cout << "\n✓ All operations completed successfully!" << std::endl;
    std::cout << "\nFiles created:" << std::endl;
    std::cout << "  - students.dat (binary data file)" << std::endl;
    std::cout << "  - students.dat.backup (backup file)" << std::endl;
    std::cout << "  - students.csv (CSV export)" << std::endl;
    std::cout << "  - grade_report.txt (formatted report)" << std::endl;
    std::cout << "  - system.log (operation log)" << std::endl;

    std::cout << "\n=== Project demonstrates: ===" << std::endl;
    std::cout << "✓ Binary file I/O" << std::endl;
    std::cout << "✓ CSV import/export" << std::endl;
    std::cout << "✓ Formatted report generation" << std::endl;
    std::cout << "✓ File backup and recovery" << std::endl;
    std::cout << "✓ Logging system" << std::endl;
    std::cout << "✓ Error handling" << std::endl;
    std::cout << "✓ Data management (CRUD operations)" << std::endl;

    return 0;
}
