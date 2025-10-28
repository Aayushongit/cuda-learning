/**
 * 04_custom_exceptions.cpp
 *
 * TOPIC: Custom Exception Classes
 *
 * This file demonstrates:
 * - Creating custom exception classes
 * - Inheriting from std::exception
 * - Adding custom data members
 * - Overriding what() method
 * - Best practices for custom exceptions
 */

#include <iostream>
#include <exception>
#include <string>
#include <sstream>

using namespace std;

// Example 1: Simple custom exception
class DatabaseException : public exception {
private:
    string message;

public:
    DatabaseException(const string& msg) : message(msg) {}

    // Override what() to provide error message
    const char* what() const noexcept override {
        return message.c_str();
    }
};

// Example 2: Custom exception with additional data
class NetworkException : public exception {
private:
    string message;
    int errorCode;
    string serverAddress;

public:
    NetworkException(const string& msg, int code, const string& address)
        : message(msg), errorCode(code), serverAddress(address) {}

    const char* what() const noexcept override {
        return message.c_str();
    }

    int getErrorCode() const { return errorCode; }
    string getServerAddress() const { return serverAddress; }
};

// Example 3: Exception hierarchy for different error types
class FileException : public exception {
protected:
    string filename;
    string message;

public:
    FileException(const string& file, const string& msg)
        : filename(file), message(msg) {}

    const char* what() const noexcept override {
        return message.c_str();
    }

    string getFilename() const { return filename; }
};

// Derived custom exceptions
class FileNotFoundException : public FileException {
public:
    FileNotFoundException(const string& file)
        : FileException(file, "File not found: " + file) {}
};

class FilePermissionException : public FileException {
public:
    FilePermissionException(const string& file)
        : FileException(file, "Permission denied: " + file) {}
};

class FileFormatException : public FileException {
private:
    int lineNumber;

public:
    FileFormatException(const string& file, int line)
        : FileException(file, "Invalid format in file: " + file + " at line " + to_string(line)),
          lineNumber(line) {}

    int getLineNumber() const { return lineNumber; }
};

// Example 4: Custom exception with detailed error context
class ValidationException : public exception {
private:
    string fieldName;
    string invalidValue;
    string reason;
    mutable string fullMessage;  // mutable because what() is const

public:
    ValidationException(const string& field, const string& value, const string& why)
        : fieldName(field), invalidValue(value), reason(why) {
        // Build detailed message
        ostringstream oss;
        oss << "Validation failed for field '" << fieldName
            << "' with value '" << invalidValue
            << "': " << reason;
        fullMessage = oss.str();
    }

    const char* what() const noexcept override {
        return fullMessage.c_str();
    }

    string getFieldName() const { return fieldName; }
    string getInvalidValue() const { return invalidValue; }
    string getReason() const { return reason; }
};

// Simulation functions that throw custom exceptions

void connectToDatabase(const string& dbName) {
    if (dbName.empty()) {
        throw DatabaseException("Database name cannot be empty!");
    }
    if (dbName == "invalid_db") {
        throw DatabaseException("Failed to connect to database: " + dbName);
    }
    cout << "   Successfully connected to database: " << dbName << endl;
}

void sendNetworkRequest(const string& server, int port) {
    if (port < 0 || port > 65535) {
        throw NetworkException("Invalid port number", 400, server);
    }
    if (server == "unreachable.com") {
        throw NetworkException("Server unreachable", 503, server);
    }
    cout << "   Request sent to " << server << ":" << port << endl;
}

void openFile(const string& filename) {
    if (filename.empty()) {
        throw FileNotFoundException("(empty)");
    }
    if (filename == "protected.txt") {
        throw FilePermissionException(filename);
    }
    if (filename == "corrupted.dat") {
        throw FileFormatException(filename, 42);
    }
    cout << "   File opened successfully: " << filename << endl;
}

void validateAge(const string& ageStr) {
    if (ageStr.empty()) {
        throw ValidationException("age", ageStr, "value cannot be empty");
    }

    // Check if it's a number
    for (char c : ageStr) {
        if (!isdigit(c)) {
            throw ValidationException("age", ageStr, "must be a numeric value");
        }
    }

    int age = stoi(ageStr);
    if (age < 0 || age > 150) {
        throw ValidationException("age", ageStr, "must be between 0 and 150");
    }

    cout << "   Age validated: " << age << endl;
}

int main() {
    cout << "=== Custom Exception Classes ===" << endl;

    // Example 1: Simple custom exception
    cout << "\n1. Simple Custom Exception:" << endl;
    try {
        connectToDatabase("");
    }
    catch (const DatabaseException& e) {
        cout << "   Caught DatabaseException: " << e.what() << endl;
    }

    try {
        connectToDatabase("invalid_db");
    }
    catch (const DatabaseException& e) {
        cout << "   Caught DatabaseException: " << e.what() << endl;
    }

    // Example 2: Exception with additional data
    cout << "\n2. Exception with Additional Data:" << endl;
    try {
        sendNetworkRequest("unreachable.com", 8080);
    }
    catch (const NetworkException& e) {
        cout << "   Caught NetworkException:" << endl;
        cout << "   - Message: " << e.what() << endl;
        cout << "   - Error Code: " << e.getErrorCode() << endl;
        cout << "   - Server: " << e.getServerAddress() << endl;
    }

    // Example 3: Exception hierarchy
    cout << "\n3. Exception Hierarchy:" << endl;

    cout << "\n   a) File not found:" << endl;
    try {
        openFile("");
    }
    catch (const FileNotFoundException& e) {
        cout << "      Caught FileNotFoundException: " << e.what() << endl;
        cout << "      Filename: " << e.getFilename() << endl;
    }

    cout << "\n   b) Permission denied:" << endl;
    try {
        openFile("protected.txt");
    }
    catch (const FilePermissionException& e) {
        cout << "      Caught FilePermissionException: " << e.what() << endl;
    }

    cout << "\n   c) Format error:" << endl;
    try {
        openFile("corrupted.dat");
    }
    catch (const FileFormatException& e) {
        cout << "      Caught FileFormatException: " << e.what() << endl;
        cout << "      Line number: " << e.getLineNumber() << endl;
    }

    cout << "\n   d) Catching by base class:" << endl;
    try {
        openFile("missing.txt");
    }
    catch (const FileException& e) {
        // Can catch all file-related exceptions
        cout << "      Caught via FileException base: " << e.what() << endl;
    }

    // Example 4: Detailed validation exception
    cout << "\n4. Validation Exception:" << endl;

    string testCases[] = {"", "abc", "25", "-5", "200"};

    for (const string& testCase : testCases) {
        cout << "\n   Testing value: '" << testCase << "'" << endl;
        try {
            validateAge(testCase);
        }
        catch (const ValidationException& e) {
            cout << "      Caught ValidationException:" << endl;
            cout << "      - Full message: " << e.what() << endl;
            cout << "      - Field: " << e.getFieldName() << endl;
            cout << "      - Invalid value: '" << e.getInvalidValue() << "'" << endl;
            cout << "      - Reason: " << e.getReason() << endl;
        }
    }

    cout << "\n=== Program completed successfully ===" << endl;

    /**
     * KEY POINTS:
     * 1. Inherit from std::exception (or derived classes)
     * 2. Override what() method with noexcept specifier
     * 3. Add custom data members for additional context
     * 4. Create exception hierarchies for related errors
     * 5. Use const char* what() const noexcept pattern
     * 6. Store error messages as class members
     * 7. Provide getter methods for additional data
     * 8. Make exceptions informative and actionable
     * 9. Use descriptive exception names (ends with Exception)
     * 10. Keep exception classes simple and focused
     */

    return 0;
}
