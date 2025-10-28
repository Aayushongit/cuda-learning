# C++ File I/O Complete Tutorial

A comprehensive, hands-on guide to file input/output operations in C++. This tutorial covers everything from basic concepts to advanced techniques through 14 well-documented example programs.

## 📚 Table of Contents

1. [Basic File Reading](#1-basic-file-reading)
2. [Basic File Writing](#2-basic-file-writing)
3. [Append Mode](#3-append-mode)
4. [Binary File I/O](#4-binary-file-io)
5. [Line-by-Line Reading](#5-line-by-line-reading)
6. [Formatted I/O](#6-formatted-io)
7. [File Positioning & Seeking](#7-file-positioning--seeking)
8. [File State Checking](#8-file-state-checking)
9. [CSV File Handling](#9-csv-file-handling)
10. [Random Access Files](#10-random-access-files)
11. [File Utilities](#11-file-utilities)
12. [StringStream with Files](#12-stringstream-with-files)
13. [Error Handling](#13-error-handling)
14. [Complete Project](#14-complete-project)

---

## Programs Overview

### 1. Basic File Reading
**File:** `01_basic_read.cpp`

Learn the fundamentals of reading files in C++.

**Key Concepts:**
- Opening files with `ifstream`
- Checking if file opened successfully
- Three methods of reading:
  - Character by character with `get()`
  - Word by word with `>>`
  - Entire file at once
- Proper file closure

**Example Output:**
```
Hello, File I/O!
This is line 2.
Learning C++ file operations.
```

**Compile & Run:**
```bash
g++ 01_basic_read.cpp -o basic_read
./basic_read
```

---

### 2. Basic File Writing
**File:** `02_basic_write.cpp`

Master the basics of writing data to files.

**Key Concepts:**
- Opening files with `ofstream`
- Write modes (overwrite vs. create)
- Writing different data types
- Formatted output to files
- Verification by reading back

**Important Modes:**
- `std::ios::out` - Open for writing (default)
- `std::ios::trunc` - Overwrite if exists (default)

**Compile & Run:**
```bash
g++ 02_basic_write.cpp -o basic_write
./basic_write
```

---

### 3. Append Mode
**File:** `03_append_mode.cpp`

Learn to add content without overwriting existing data.

**Key Concepts:**
- `std::ios::app` mode
- Difference between append and overwrite
- Logging pattern implementation
- Timestamp formatting
- Preserving existing content

**Practical Use Cases:**
- Application logs
- Data accumulation
- Incremental file updates

**Compile & Run:**
```bash
g++ 03_append_mode.cpp -o append_mode
./append_mode
```

---

### 4. Binary File I/O
**File:** `04_binary_io.cpp`

Work with binary data for efficient storage.

**Key Concepts:**
- `std::ios::binary` flag
- `read()` and `write()` methods
- Storing structs/objects
- Binary vs. text storage comparison
- `reinterpret_cast` usage

**Advantages of Binary Files:**
- Smaller file sizes
- Faster I/O operations
- Exact data representation
- No conversion overhead

**Compile & Run:**
```bash
g++ 04_binary_io.cpp -o binary_io
./binary_io
```

---

### 5. Line-by-Line Reading
**File:** `05_line_by_line.cpp`

Process text files line by line efficiently.

**Key Concepts:**
- `getline()` function
- Custom delimiters
- Line filtering and searching
- Word counting
- Reverse file reading
- Storing lines in vectors

**Six Different Methods:**
1. Basic line-by-line reading
2. Storing in vector
3. Processing (word count)
4. Filtering with search
5. Custom delimiter
6. Reverse order reading

**Compile & Run:**
```bash
g++ 05_line_by_line.cpp -o line_by_line
./line_by_line
```

---

### 6. Formatted I/O
**File:** `06_formatted_io.cpp`

Create beautifully formatted output files.

**Key Concepts:**
- I/O manipulators (`setw`, `setprecision`, `setfill`)
- Field width and alignment
- Number formatting (decimal, hex, octal)
- Boolean formatting (`boolalpha`)
- Creating tables and reports
- Box drawing characters

**Important Manipulators:**
- `std::left` / `std::right` - Alignment
- `std::setw(n)` - Field width
- `std::setprecision(n)` - Decimal places
- `std::fixed` - Fixed-point notation
- `std::scientific` - Scientific notation

**Compile & Run:**
```bash
g++ 06_formatted_io.cpp -o formatted_io
./formatted_io
```

---

### 7. File Positioning & Seeking
**File:** `07_seeking_position.cpp`

Navigate and access any part of a file.

**Key Concepts:**
- `tellg()` / `tellp()` - Get position
- `seekg()` / `seekp()` - Set position
- Position origins:
  - `std::ios::beg` - Beginning
  - `std::ios::cur` - Current position
  - `std::ios::end` - End of file
- Random access patterns
- Reading files in reverse
- Modifying specific locations

**Practical Applications:**
- Editing config files
- Random access databases
- File indexing
- Partial file updates

**Compile & Run:**
```bash
g++ 07_seeking_position.cpp -o seeking_position
./seeking_position
```

---

### 8. File State Checking
**File:** `08_file_state.cpp`

Robust error detection and handling.

**Key Concepts:**
- `is_open()` - File opened?
- `good()` - Stream ready?
- `eof()` - End of file?
- `fail()` - Logical error?
- `bad()` - Fatal error?
- `clear()` - Reset error flags
- Exception handling with streams

**State Checking Pattern:**
```cpp
if (!file.is_open()) { /* handle */ }
if (!file.good()) { /* handle */ }
while (file >> data && file.good()) { /* process */ }
if (file.eof()) { /* normal end */ }
else if (file.fail()) { /* error */ }
```

**Compile & Run:**
```bash
g++ 08_file_state.cpp -o file_state
./file_state
```

---

### 9. CSV File Handling
**File:** `09_csv_handling.cpp`

Work with comma-separated values files.

**Key Concepts:**
- Reading CSV files
- Writing CSV files
- Parsing CSV data
- Handling quoted fields
- Data filtering
- Data aggregation
- Creating summary reports

**Eight Techniques:**
1. Writing CSV
2. Reading and parsing
3. Alternative parsing with stringstream
4. Formatted CSV writing
5. Data filtering
6. Handling quoted fields
7. Data aggregation
8. Creating summaries

**Compile & Run:**
```bash
g++ 09_csv_handling.cpp -o csv_handling
./csv_handling
```

---

### 10. Random Access Files
**File:** `10_random_access.cpp`

Build database-like file systems.

**Key Concepts:**
- Fixed-size records
- Direct positioning
- Read/write at any position
- Building a record database
- Sparse files
- In-place modifications

**Database Class Features:**
- `writeRecord()` - Write at position
- `readRecord()` - Read from position
- `updateRecord()` - Modify existing
- `getRecordCount()` - Count records
- `displayAll()` - Show all records

**Compile & Run:**
```bash
g++ 10_random_access.cpp -o random_access
./random_access
```

---

### 11. File Utilities
**File:** `11_file_utilities.cpp`

Essential file manipulation operations.

**Key Concepts:**
- File existence checking
- Getting file size
- Copying files
- Comparing files
- Counting lines/words
- File modification time
- Reading last N lines
- Merging files
- Temporary files

**Utility Functions:**
```cpp
bool fileExists(const string& filename);
long getFileSize(const string& filename);
bool copyFile(const string& src, const string& dst);
bool filesAreIdentical(const string& f1, const string& f2);
int countLines(const string& filename);
int countWords(const string& filename);
```

**Compile & Run:**
```bash
g++ 11_file_utilities.cpp -o file_utilities
./file_utilities
```

---

### 12. StringStream with Files
**File:** `12_stringstream_file.cpp`

Combine in-memory string processing with file I/O.

**Key Concepts:**
- Using `stringstream` for parsing
- Building formatted strings
- Type conversions
- Processing complex data formats
- CSV parsing with streams
- Reusing stringstreams
- JSON-like output generation

**StringStream Operations:**
```cpp
stringstream ss;
ss << data;              // Write to stream
ss >> variable;          // Read from stream
string result = ss.str(); // Get string
ss.str("");              // Clear content
ss.clear();              // Clear state flags
```

**Compile & Run:**
```bash
g++ 12_stringstream_file.cpp -o stringstream_file
./stringstream_file
```

---

### 13. Error Handling
**File:** `13_error_handling.cpp`

Best practices for robust file operations.

**Key Concepts:**
- RAII pattern
- Exception handling
- Comprehensive error checking
- Backup before modification
- Error recovery
- Safe file operations
- Transaction-like updates

**Best Practices:**
1. Always check if file opened
2. Use RAII for automatic cleanup
3. Check stream state after operations
4. Use try-catch blocks
5. Create backups for critical files
6. Verify writes when necessary
7. Close files explicitly
8. Handle both logical and fatal errors

**RAII Pattern:**
```cpp
class SafeFile {
public:
    SafeFile(const string& name, ios::openmode mode);
    ~SafeFile(); // Automatic cleanup
};
```

**Compile & Run:**
```bash
g++ 13_error_handling.cpp -o error_handling
./error_handling
```

---

### 14. Complete Project
**File:** `14_complete_project.cpp`

**Student Grade Management System** - A real-world application combining all concepts.

**Features:**
- Binary data storage
- CSV import/export
- Formatted report generation
- Backup and recovery
- Operation logging
- Full CRUD operations
- Statistics calculation

**System Capabilities:**
1. Add students
2. Update grades
3. Delete students (soft delete)
4. Display all students
5. Export to CSV
6. Import from CSV
7. Generate formatted reports
8. Create backups
9. Automatic logging

**Files Created:**
- `students.dat` - Binary data file
- `students.dat.backup` - Backup file
- `students.csv` - CSV export
- `grade_report.txt` - Formatted report
- `system.log` - Operation log

**Compile & Run:**
```bash
g++ 14_complete_project.cpp -o grade_system
./grade_system
```

---

## 🔧 Compilation Guide

### Compile All Programs
```bash
# Individual compilation
g++ -std=c++11 01_basic_read.cpp -o 01_basic_read
g++ -std=c++11 02_basic_write.cpp -o 02_basic_write
# ... and so on

# Or compile all at once
for file in *.cpp; do
    g++ -std=c++11 "$file" -o "${file%.cpp}"
done
```

### Run All Programs
```bash
# Run each program
./01_basic_read
./02_basic_write
./03_append_mode
# ... and so on
```

---

## 📖 Learning Path

### Beginner Level (Start Here)
1. **01_basic_read.cpp** - Reading files
2. **02_basic_write.cpp** - Writing files
3. **03_append_mode.cpp** - Appending data
4. **05_line_by_line.cpp** - Processing lines

### Intermediate Level
5. **06_formatted_io.cpp** - Formatting output
6. **08_file_state.cpp** - Error detection
7. **09_csv_handling.cpp** - CSV files
8. **12_stringstream_file.cpp** - String processing

### Advanced Level
9. **04_binary_io.cpp** - Binary operations
10. **07_seeking_position.cpp** - Random access
11. **10_random_access.cpp** - Database-like files
12. **11_file_utilities.cpp** - File operations
13. **13_error_handling.cpp** - Robust code

### Project Level
14. **14_complete_project.cpp** - Complete application

---

## 🎯 Key Takeaways

### File Streams
```cpp
ifstream  // Input file stream (reading)
ofstream  // Output file stream (writing)
fstream   // File stream (both reading and writing)
```

### Open Modes
```cpp
ios::in       // Read
ios::out      // Write
ios::app      // Append
ios::ate      // At end
ios::trunc    // Truncate
ios::binary   // Binary mode
```

### Position Functions
```cpp
tellg()  // Get read position
tellp()  // Get write position
seekg()  // Set read position
seekp()  // Set write position
```

### State Functions
```cpp
is_open()  // File opened?
good()     // Ready for I/O?
eof()      // End of file?
fail()     // Logical error?
bad()      // Fatal error?
clear()    // Clear error flags
```

---

## 💡 Common Patterns

### Safe File Reading
```cpp
ifstream file("data.txt");
if (!file.is_open()) {
    cerr << "Error opening file" << endl;
    return;
}

string line;
while (getline(file, line)) {
    // Process line
}

file.close();
```

### Safe File Writing
```cpp
ofstream file("output.txt");
if (!file.is_open()) {
    cerr << "Error creating file" << endl;
    return;
}

file << "Data to write" << endl;
file.close();
```

### Binary I/O
```cpp
// Writing
ofstream file("data.bin", ios::binary);
file.write(reinterpret_cast<char*>(&data), sizeof(data));

// Reading
ifstream file("data.bin", ios::binary);
file.read(reinterpret_cast<char*>(&data), sizeof(data));
```

---

## 🐛 Common Pitfalls

1. **Forgetting to check if file opened**
   ```cpp
   // BAD
   ifstream file("data.txt");
   file >> data;  // May fail!

   // GOOD
   ifstream file("data.txt");
   if (file.is_open()) {
       file >> data;
   }
   ```

2. **Not closing files explicitly**
   ```cpp
   // Use RAII or close explicitly
   file.close();
   ```

3. **Mixing text and binary modes**
   ```cpp
   // Don't mix these!
   file << data;           // Text mode
   file.write(...);        // Binary mode
   ```

4. **Ignoring error states**
   ```cpp
   // Always check!
   if (file.fail()) { /* handle error */ }
   ```

---

## 🚀 Next Steps

1. **Practice:** Run each program and study the output
2. **Modify:** Change parameters and see what happens
3. **Extend:** Add features to the examples
4. **Build:** Create your own file-based applications

### Project Ideas
- **Todo List Manager** - Save tasks to file
- **Contact Book** - Store contacts in binary format
- **Log Analyzer** - Parse and analyze log files
- **Config Manager** - Read/write configuration files
- **Simple Database** - Implement a file-based database

---

## 📚 Additional Resources

### C++ Documentation
- [cppreference.com - File I/O](https://en.cppreference.com/w/cpp/io)
- [C++ fstream Reference](https://cplusplus.com/reference/fstream/)

### Books
- "C++ Primer" by Lippman, Lajoie, and Moo
- "Effective C++" by Scott Meyers

---

## ✅ Checklist

After completing this tutorial, you should be able to:

- [ ] Open and close files safely
- [ ] Read files using multiple methods
- [ ] Write formatted data to files
- [ ] Append data without overwriting
- [ ] Work with binary files
- [ ] Process files line by line
- [ ] Navigate files using seek operations
- [ ] Check and handle file errors
- [ ] Parse CSV files
- [ ] Implement random access
- [ ] Use file utility functions
- [ ] Combine stringstream with files
- [ ] Apply error handling best practices
- [ ] Build complete file-based applications

---

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Add more examples
- Fix bugs

---

## 📝 License

This tutorial is provided as educational material. Feel free to use and modify for learning purposes.

---

## 🎓 Conclusion

File I/O is a fundamental skill in C++ programming. These 14 programs provide a complete learning path from basics to advanced techniques. Practice each example, understand the concepts, and apply them to your own projects.

**Happy Coding! 🚀**

---

*Last Updated: 2024*
