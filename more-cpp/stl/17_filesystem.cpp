/**
 * 17_filesystem.cpp
 *
 * FILESYSTEM OPERATIONS (C++17)
 * - Path manipulation
 * - Directory operations
 * - File operations
 * - File queries
 * - Directory iteration
 */

#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== FILESYSTEM OPERATIONS ===\n";

    separator("PATH OPERATIONS");

    // 1. Path Construction
    std::cout << "\n1. PATH CONSTRUCTION:\n";
    fs::path p1 = "/home/user/documents";
    fs::path p2 = "file.txt";
    fs::path p3 = p1 / p2;  // Concatenate paths

    std::cout << "p1: " << p1 << "\n";
    std::cout << "p2: " << p2 << "\n";
    std::cout << "p3: " << p3 << "\n";

    // 2. Path Components
    std::cout << "\n2. PATH COMPONENTS:\n";
    fs::path full_path = "/home/user/documents/file.txt";

    std::cout << "Full path: " << full_path << "\n";
    std::cout << "Root: " << full_path.root_path() << "\n";
    std::cout << "Parent: " << full_path.parent_path() << "\n";
    std::cout << "Filename: " << full_path.filename() << "\n";
    std::cout << "Stem: " << full_path.stem() << "\n";
    std::cout << "Extension: " << full_path.extension() << "\n";

    // 3. Path Manipulation
    std::cout << "\n3. PATH MANIPULATION:\n";
    fs::path doc_path = "document.txt";

    std::cout << "Original: " << doc_path << "\n";

    doc_path.replace_extension(".pdf");
    std::cout << "After replace_extension: " << doc_path << "\n";

    doc_path.replace_filename("newfile.pdf");
    std::cout << "After replace_filename: " << doc_path << "\n";

    // 4. Path Queries
    std::cout << "\n4. PATH QUERIES:\n";
    fs::path test_path = "/home/user/../user/./documents";

    std::cout << "Has root: " << (test_path.has_root_path() ? "yes" : "no") << "\n";
    std::cout << "Has parent: " << (test_path.has_parent_path() ? "yes" : "no") << "\n";
    std::cout << "Has filename: " << (test_path.has_filename() ? "yes" : "no") << "\n";
    std::cout << "Has extension: " << (test_path.has_extension() ? "yes" : "no") << "\n";
    std::cout << "Is absolute: " << (test_path.is_absolute() ? "yes" : "no") << "\n";
    std::cout << "Is relative: " << (test_path.is_relative() ? "yes" : "no") << "\n";

    separator("DIRECTORY OPERATIONS");

    // 5. Create Directory
    std::cout << "\n5. CREATE DIRECTORY:\n";
    fs::path test_dir = "test_directory";

    if (!fs::exists(test_dir)) {
        fs::create_directory(test_dir);
        std::cout << "Created: " << test_dir << "\n";
    } else {
        std::cout << "Directory already exists\n";
    }

    // 6. Create Nested Directories
    std::cout << "\n6. CREATE NESTED DIRECTORIES:\n";
    fs::path nested = "test_directory/level1/level2";

    fs::create_directories(nested);
    std::cout << "Created nested path: " << nested << "\n";

    // 7. Current Path
    std::cout << "\n7. CURRENT PATH:\n";
    std::cout << "Current directory: " << fs::current_path() << "\n";

    // 8. Temporary Directory
    std::cout << "\n8. TEMPORARY DIRECTORY:\n";
    std::cout << "Temp directory: " << fs::temp_directory_path() << "\n";

    separator("FILE OPERATIONS");

    // 9. Create and Write File
    std::cout << "\n9. CREATE FILE:\n";
    fs::path test_file = test_dir / "sample.txt";
    {
        std::ofstream ofs(test_file);
        ofs << "Hello, Filesystem!\n";
    }
    std::cout << "Created file: " << test_file << "\n";

    // 10. Copy File
    std::cout << "\n10. COPY FILE:\n";
    fs::path copy_dest = test_dir / "sample_copy.txt";
    fs::copy_file(test_file, copy_dest, fs::copy_options::overwrite_existing);
    std::cout << "Copied to: " << copy_dest << "\n";

    // 11. Rename/Move
    std::cout << "\n11. RENAME FILE:\n";
    fs::path renamed = test_dir / "renamed.txt";
    fs::rename(copy_dest, renamed);
    std::cout << "Renamed to: " << renamed << "\n";

    separator("FILE QUERIES");

    // 12. File Existence
    std::cout << "\n12. FILE EXISTENCE:\n";
    std::cout << test_file << " exists: " << (fs::exists(test_file) ? "yes" : "no") << "\n";
    std::cout << "nonexistent.txt exists: " << (fs::exists("nonexistent.txt") ? "yes" : "no") << "\n";

    // 13. File Type
    std::cout << "\n13. FILE TYPE:\n";
    std::cout << test_file << " is regular file: " << (fs::is_regular_file(test_file) ? "yes" : "no") << "\n";
    std::cout << test_dir << " is directory: " << (fs::is_directory(test_dir) ? "yes" : "no") << "\n";

    // 14. File Size
    std::cout << "\n14. FILE SIZE:\n";
    if (fs::exists(test_file)) {
        std::cout << "Size of " << test_file.filename() << ": " << fs::file_size(test_file) << " bytes\n";
    }

    // 15. File Timestamps
    std::cout << "\n15. FILE TIMESTAMPS:\n";
    if (fs::exists(test_file)) {
        auto ftime = fs::last_write_time(test_file);
        std::cout << "Last modified time recorded\n";

        // Modify timestamp
        auto new_time = ftime + std::chrono::hours(24);
        fs::last_write_time(test_file, new_time);
        std::cout << "Modified timestamp\n";
    }

    separator("DIRECTORY ITERATION");

    // 16. List Directory Contents
    std::cout << "\n16. LIST DIRECTORY:\n";
    std::cout << "Contents of " << test_dir << ":\n";
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        std::cout << "  " << entry.path().filename() << "\n";
    }

    // 17. Recursive Iteration
    std::cout << "\n17. RECURSIVE LISTING:\n";
    std::cout << "Recursive contents:\n";
    for (const auto& entry : fs::recursive_directory_iterator(test_dir)) {
        std::cout << "  " << entry.path() << "\n";
    }

    // 18. Filter by Extension
    std::cout << "\n18. FILTER BY EXTENSION:\n";
    // Create some test files
    {
        std::ofstream(test_dir / "file1.txt");
        std::ofstream(test_dir / "file2.cpp");
        std::ofstream(test_dir / "file3.txt");
    }

    std::cout << "Text files in " << test_dir << ":\n";
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.path().extension() == ".txt") {
            std::cout << "  " << entry.path().filename() << "\n";
        }
    }

    separator("FILE PERMISSIONS");

    // 19. File Permissions
    std::cout << "\n19. FILE PERMISSIONS:\n";
    if (fs::exists(test_file)) {
        auto perms = fs::status(test_file).permissions();
        std::cout << "Permissions for " << test_file.filename() << ":\n";
        std::cout << "  Owner can read: " << ((perms & fs::perms::owner_read) != fs::perms::none ? "yes" : "no") << "\n";
        std::cout << "  Owner can write: " << ((perms & fs::perms::owner_write) != fs::perms::none ? "yes" : "no") << "\n";
    }

    separator("SPACE INFO");

    // 20. Disk Space
    std::cout << "\n20. DISK SPACE:\n";
    fs::space_info si = fs::space(fs::current_path());
    std::cout << "Disk space info:\n";
    std::cout << "  Capacity: " << (si.capacity / 1024 / 1024 / 1024) << " GB\n";
    std::cout << "  Free: " << (si.free / 1024 / 1024 / 1024) << " GB\n";
    std::cout << "  Available: " << (si.available / 1024 / 1024 / 1024) << " GB\n";

    separator("CLEANUP");

    // 21. Remove Files
    std::cout << "\n21. CLEANUP:\n";
    if (fs::exists(test_dir)) {
        fs::remove_all(test_dir);
        std::cout << "Removed test directory and all contents\n";
    }

    separator("PRACTICAL EXAMPLES");

    // 22. Find Files by Name
    std::cout << "\n22. FIND FILES BY NAME:\n";
    auto find_files = [](const fs::path& dir, const std::string& pattern) {
        std::cout << "Searching for '*" << pattern << "' in " << dir << ":\n";
        if (fs::exists(dir) && fs::is_directory(dir)) {
            for (const auto& entry : fs::recursive_directory_iterator(dir)) {
                if (entry.is_regular_file() &&
                    entry.path().filename().string().find(pattern) != std::string::npos) {
                    std::cout << "  Found: " << entry.path() << "\n";
                }
            }
        }
    };

    // Create test structure
    fs::create_directories("search_test/subdir");
    std::ofstream("search_test/test1.txt");
    std::ofstream("search_test/test2.cpp");
    std::ofstream("search_test/subdir/test3.txt");

    find_files("search_test", "test");

    // Cleanup
    fs::remove_all("search_test");

    // 23. Calculate Directory Size
    std::cout << "\n23. CALCULATE DIRECTORY SIZE:\n";
    auto calculate_size = [](const fs::path& dir) -> uintmax_t {
        uintmax_t total = 0;
        if (fs::exists(dir) && fs::is_directory(dir)) {
            for (const auto& entry : fs::recursive_directory_iterator(dir)) {
                if (entry.is_regular_file()) {
                    total += fs::file_size(entry);
                }
            }
        }
        return total;
    };

    // Create test
    fs::create_directory("size_test");
    {
        std::ofstream ofs("size_test/file.txt");
        ofs << std::string(1024, 'x');  // 1KB
    }

    std::cout << "Size of size_test: " << calculate_size("size_test") << " bytes\n";

    // Cleanup
    fs::remove_all("size_test");

    std::cout << "\n=== END OF FILESYSTEM ===\n";

    return 0;
}
