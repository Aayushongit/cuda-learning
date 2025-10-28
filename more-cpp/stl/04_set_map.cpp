/**
 * 04_set_map.cpp
 *
 * ASSOCIATIVE CONTAINERS (Ordered, Tree-based - typically Red-Black Trees)
 * - set: Unique sorted elements
 * - multiset: Sorted elements with duplicates allowed
 * - map: Key-value pairs, unique keys, sorted by key
 * - multimap: Key-value pairs, duplicate keys allowed, sorted
 *
 * Common Features:
 * - Automatic sorting
 * - Logarithmic time complexity O(log n) for insert/delete/find
 * - Bidirectional iterators
 * - Keys are immutable
 */

#include <iostream>
#include <set>
#include <map>
#include <string>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== ASSOCIATIVE CONTAINERS: SET & MAP ===\n";

    // ========== STD::SET ==========
    separator("STD::SET");

    // 1. Set Initialization
    std::cout << "\n1. SET INITIALIZATION:\n";
    std::set<int> set1;                         // Empty set
    std::set<int> set2 = {5, 2, 8, 1, 9, 3};   // Initializer list (auto-sorted)
    std::set<int> set3(set2);                   // Copy constructor

    std::cout << "set2 (auto-sorted): ";
    for (const auto& val : set2) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // Custom comparator (descending order)
    std::set<int, std::greater<int>> desc_set = {5, 2, 8, 1, 9};
    std::cout << "Descending set: ";
    for (const auto& val : desc_set) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // 2. Inserting Elements
    std::cout << "\n2. INSERTING INTO SET:\n";
    std::set<int> numbers;

    // insert() returns pair<iterator, bool>
    auto [it1, success1] = numbers.insert(10);
    std::cout << "Insert 10: " << (success1 ? "success" : "failed") << "\n";

    auto [it2, success2] = numbers.insert(10);  // Duplicate
    std::cout << "Insert 10 again: " << (success2 ? "success" : "failed") << "\n";

    numbers.insert(20);
    numbers.insert(5);
    numbers.insert(15);

    std::cout << "Set: ";
    for (const auto& n : numbers) std::cout << n << " ";
    std::cout << "\n";

    // Insert multiple
    numbers.insert({25, 30, 35});
    std::cout << "After inserting {25, 30, 35}: ";
    for (const auto& n : numbers) std::cout << n << " ";
    std::cout << "\n";

    // Emplace (construct in-place)
    numbers.emplace(12);
    std::cout << "After emplace(12): ";
    for (const auto& n : numbers) std::cout << n << " ";
    std::cout << "\n";

    // 3. Finding Elements
    std::cout << "\n3. FINDING IN SET:\n";
    std::set<int> search_set = {1, 3, 5, 7, 9, 11};

    // find() returns iterator
    auto find_it = search_set.find(5);
    if (find_it != search_set.end()) {
        std::cout << "Found: " << *find_it << "\n";
    }

    // count() returns 0 or 1 for set
    std::cout << "count(7) = " << search_set.count(7) << "\n";
    std::cout << "count(100) = " << search_set.count(100) << "\n";

    // contains() (C++20)
    std::cout << "contains(9) = " << (search_set.contains(9) ? "true" : "false") << "\n";

    // lower_bound and upper_bound
    auto lb = search_set.lower_bound(5);  // First element >= 5
    auto ub = search_set.upper_bound(5);  // First element > 5
    std::cout << "lower_bound(5) = " << *lb << "\n";
    std::cout << "upper_bound(5) = " << *ub << "\n";

    // equal_range
    auto [lower, upper] = search_set.equal_range(5);
    std::cout << "equal_range(5): [" << *lower << ", " << *upper << ")\n";

    // 4. Removing Elements
    std::cout << "\n4. REMOVING FROM SET:\n";
    std::set<int> remove_set = {1, 2, 3, 4, 5, 6, 7, 8};

    std::cout << "Original: ";
    for (const auto& n : remove_set) std::cout << n << " ";
    std::cout << "\n";

    remove_set.erase(5);  // Erase by value
    std::cout << "After erase(5): ";
    for (const auto& n : remove_set) std::cout << n << " ";
    std::cout << "\n";

    auto erase_it = remove_set.find(3);
    if (erase_it != remove_set.end()) {
        remove_set.erase(erase_it);  // Erase by iterator
    }
    std::cout << "After erase(iterator to 3): ";
    for (const auto& n : remove_set) std::cout << n << " ";
    std::cout << "\n";

    // 5. Set Operations
    std::cout << "\n5. SET OPERATIONS:\n";
    std::set<int> a = {1, 2, 3, 4, 5};
    std::set<int> b = {4, 5, 6, 7, 8};

    std::cout << "Set A: ";
    for (const auto& n : a) std::cout << n << " ";
    std::cout << "\n";

    std::cout << "Set B: ";
    for (const auto& n : b) std::cout << n << " ";
    std::cout << "\n";

    std::cout << "size() = " << a.size() << "\n";
    std::cout << "empty() = " << (a.empty() ? "true" : "false") << "\n";

    // ========== STD::MULTISET ==========
    separator("STD::MULTISET");

    // 6. Multiset (allows duplicates)
    std::cout << "\n6. MULTISET (DUPLICATES ALLOWED):\n";
    std::multiset<int> mset = {5, 2, 8, 2, 9, 5, 1, 5};

    std::cout << "Multiset: ";
    for (const auto& n : mset) std::cout << n << " ";
    std::cout << "\n";

    std::cout << "count(5) = " << mset.count(5) << "\n";
    std::cout << "count(2) = " << mset.count(2) << "\n";

    mset.insert(5);
    std::cout << "After insert(5), count(5) = " << mset.count(5) << "\n";

    // Erase all occurrences
    mset.erase(5);
    std::cout << "After erase(5): ";
    for (const auto& n : mset) std::cout << n << " ";
    std::cout << "\n";

    // ========== STD::MAP ==========
    separator("STD::MAP");

    // 7. Map Initialization
    std::cout << "\n7. MAP INITIALIZATION:\n";
    std::map<std::string, int> ages;                              // Empty
    std::map<std::string, int> scores = {
        {"Alice", 95},
        {"Bob", 87},
        {"Charlie", 92}
    };

    std::cout << "Scores:\n";
    for (const auto& [name, score] : scores) {
        std::cout << name << ": " << score << "\n";
    }

    // 8. Inserting into Map
    std::cout << "\n8. INSERTING INTO MAP:\n";
    std::map<int, std::string> students;

    // Method 1: insert with pair
    students.insert(std::pair<int, std::string>(1, "John"));

    // Method 2: insert with make_pair
    students.insert(std::make_pair(2, "Jane"));

    // Method 3: insert with initializer list
    students.insert({3, "Jack"});

    // Method 4: operator[] (creates if doesn't exist)
    students[4] = "Jill";

    // Method 5: emplace (construct in-place)
    students.emplace(5, "Joe");

    std::cout << "Students:\n";
    for (const auto& [id, name] : students) {
        std::cout << id << ": " << name << "\n";
    }

    // Attempt to insert duplicate key
    auto [iter, inserted] = students.insert({1, "Johnny"});
    std::cout << "\nInsert duplicate key 1: " << (inserted ? "success" : "failed") << "\n";
    std::cout << "Value at key 1: " << students[1] << "\n";

    // 9. Accessing Map Elements
    std::cout << "\n9. ACCESSING MAP:\n";
    std::map<std::string, int> phone_book = {
        {"Alice", 1234},
        {"Bob", 5678},
        {"Charlie", 9012}
    };

    // operator[] - creates if doesn't exist
    std::cout << "Alice's number: " << phone_book["Alice"] << "\n";
    std::cout << "David's number: " << phone_book["David"] << "\n";  // Creates entry!

    std::cout << "After accessing David (size = " << phone_book.size() << "):\n";
    for (const auto& [name, number] : phone_book) {
        std::cout << name << ": " << number << "\n";
    }

    // at() - throws if doesn't exist
    try {
        std::cout << "\nBob's number (via at): " << phone_book.at("Bob") << "\n";
        std::cout << "Eve's number (via at): " << phone_book.at("Eve") << "\n";
    } catch (const std::out_of_range& e) {
        std::cout << "Exception: " << e.what() << "\n";
    }

    // 10. Finding in Map
    std::cout << "\n10. FINDING IN MAP:\n";
    std::map<std::string, double> prices = {
        {"apple", 1.20},
        {"banana", 0.80},
        {"orange", 1.50}
    };

    auto find_iter = prices.find("banana");
    if (find_iter != prices.end()) {
        std::cout << "Found: " << find_iter->first << " = $" << find_iter->second << "\n";
    }

    std::cout << "count('apple') = " << prices.count("apple") << "\n";
    std::cout << "contains('grape') = " << (prices.contains("grape") ? "true" : "false") << "\n";

    // 11. Modifying Map Values
    std::cout << "\n11. MODIFYING MAP VALUES:\n";
    std::map<std::string, int> inventory = {
        {"apples", 10},
        {"bananas", 5},
        {"oranges", 8}
    };

    std::cout << "Original inventory:\n";
    for (const auto& [item, count] : inventory) {
        std::cout << item << ": " << count << "\n";
    }

    inventory["apples"] += 5;  // Increase
    inventory["bananas"] = 20;  // Replace
    inventory["grapes"] = 15;   // Add new

    std::cout << "\nUpdated inventory:\n";
    for (const auto& [item, count] : inventory) {
        std::cout << item << ": " << count << "\n";
    }

    // 12. Removing from Map
    std::cout << "\n12. REMOVING FROM MAP:\n";
    std::map<int, std::string> items = {
        {1, "One"},
        {2, "Two"},
        {3, "Three"},
        {4, "Four"}
    };

    items.erase(2);  // Erase by key
    std::cout << "After erase(2):\n";
    for (const auto& [k, v] : items) {
        std::cout << k << ": " << v << "\n";
    }

    // ========== STD::MULTIMAP ==========
    separator("STD::MULTIMAP");

    // 13. Multimap (duplicate keys allowed)
    std::cout << "\n13. MULTIMAP:\n";
    std::multimap<std::string, int> grades = {
        {"Math", 90},
        {"English", 85},
        {"Math", 95},      // Duplicate key
        {"Science", 88},
        {"Math", 92}       // Duplicate key
    };

    std::cout << "All grades:\n";
    for (const auto& [subject, grade] : grades) {
        std::cout << subject << ": " << grade << "\n";
    }

    std::cout << "\ncount('Math') = " << grades.count("Math") << "\n";

    // Find all Math grades
    std::cout << "All Math grades: ";
    auto range = grades.equal_range("Math");
    for (auto it = range.first; it != range.second; ++it) {
        std::cout << it->second << " ";
    }
    std::cout << "\n";

    // 14. Custom Comparator
    std::cout << "\n14. CUSTOM COMPARATOR:\n";

    // Case-insensitive string comparison
    struct CaseInsensitiveCompare {
        bool operator()(const std::string& a, const std::string& b) const {
            return std::lexicographical_compare(
                a.begin(), a.end(),
                b.begin(), b.end(),
                [](char c1, char c2) { return std::tolower(c1) < std::tolower(c2); }
            );
        }
    };

    std::map<std::string, int, CaseInsensitiveCompare> case_map;
    case_map["Apple"] = 1;
    case_map["banana"] = 2;
    case_map["CHERRY"] = 3;

    std::cout << "Case-insensitive map:\n";
    for (const auto& [key, value] : case_map) {
        std::cout << key << ": " << value << "\n";
    }

    // 15. Performance Characteristics
    separator("WHEN TO USE WHAT");

    std::cout << "\nUse SET when:\n";
    std::cout << "- Need unique sorted elements\n";
    std::cout << "- Frequent lookups required\n";
    std::cout << "- Order matters\n";

    std::cout << "\nUse MULTISET when:\n";
    std::cout << "- Need sorted elements with duplicates\n";
    std::cout << "- Maintain priority queues with duplicates\n";

    std::cout << "\nUse MAP when:\n";
    std::cout << "- Key-value associations needed\n";
    std::cout << "- Keys must be unique\n";
    std::cout << "- Need sorted keys\n";

    std::cout << "\nUse MULTIMAP when:\n";
    std::cout << "- One key maps to multiple values\n";
    std::cout << "- Example: student -> grades mapping\n";

    std::cout << "\nAll have O(log n) complexity for:\n";
    std::cout << "- Insert, Delete, Find\n";

    std::cout << "\n=== END OF SET & MAP ===\n";

    return 0;
}
