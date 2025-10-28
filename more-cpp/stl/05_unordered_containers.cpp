/**
 * 05_unordered_containers.cpp
 *
 * UNORDERED ASSOCIATIVE CONTAINERS (Hash-based)
 * - unordered_set: Unique elements, no order
 * - unordered_multiset: Duplicates allowed, no order
 * - unordered_map: Key-value pairs, unique keys, no order
 * - unordered_multimap: Duplicate keys allowed, no order
 *
 * Features:
 * - Hash table implementation
 * - Average O(1) for insert/delete/find
 * - Worst case O(n) if many hash collisions
 * - No ordering guarantees
 * - Faster than ordered containers for lookups
 */

#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <vector>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== UNORDERED CONTAINERS ===\n";

    // ========== UNORDERED_SET ==========
    separator("UNORDERED_SET");

    // 1. Initialization
    std::cout << "\n1. UNORDERED_SET INITIALIZATION:\n";
    std::unordered_set<int> uset1;                          // Empty
    std::unordered_set<int> uset2 = {5, 2, 8, 1, 9, 3};    // Initializer list
    std::unordered_set<int> uset3(uset2);                   // Copy

    std::cout << "uset2 (no guaranteed order): ";
    for (const auto& val : uset2) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // 2. Inserting Elements
    std::cout << "\n2. INSERTING:\n";
    std::unordered_set<std::string> words;

    auto [it1, success1] = words.insert("hello");
    std::cout << "Insert 'hello': " << (success1 ? "success" : "failed") << "\n";

    auto [it2, success2] = words.insert("hello");  // Duplicate
    std::cout << "Insert 'hello' again: " << (success2 ? "success" : "failed") << "\n";

    words.insert("world");
    words.insert("cpp");
    words.emplace("programming");

    std::cout << "Words: ";
    for (const auto& word : words) {
        std::cout << word << " ";
    }
    std::cout << "\n";

    // 3. Finding Elements
    std::cout << "\n3. FINDING ELEMENTS:\n";
    std::unordered_set<int> numbers = {10, 20, 30, 40, 50};

    auto find_it = numbers.find(30);
    if (find_it != numbers.end()) {
        std::cout << "Found: " << *find_it << "\n";
    }

    std::cout << "count(40) = " << numbers.count(40) << "\n";
    std::cout << "count(100) = " << numbers.count(100) << "\n";

    std::cout << "contains(20) = " << (numbers.contains(20) ? "true" : "false") << "\n";
    std::cout << "contains(60) = " << (numbers.contains(60) ? "true" : "false") << "\n";

    // 4. Removing Elements
    std::cout << "\n4. REMOVING ELEMENTS:\n";
    std::unordered_set<int> remove_set = {1, 2, 3, 4, 5, 6};

    std::cout << "Original size: " << remove_set.size() << "\n";

    remove_set.erase(3);  // Erase by value
    std::cout << "After erase(3), size: " << remove_set.size() << "\n";

    auto erase_it = remove_set.find(5);
    if (erase_it != remove_set.end()) {
        remove_set.erase(erase_it);  // Erase by iterator
    }
    std::cout << "After erasing 5, size: " << remove_set.size() << "\n";

    // 5. Hash Function Properties
    std::cout << "\n5. HASH PROPERTIES:\n";
    std::unordered_set<int> hash_set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::cout << "size() = " << hash_set.size() << "\n";
    std::cout << "bucket_count() = " << hash_set.bucket_count() << "\n";
    std::cout << "load_factor() = " << hash_set.load_factor() << "\n";
    std::cout << "max_load_factor() = " << hash_set.max_load_factor() << "\n";

    // Reserve buckets
    hash_set.reserve(100);
    std::cout << "\nAfter reserve(100):\n";
    std::cout << "bucket_count() = " << hash_set.bucket_count() << "\n";

    // Bucket information
    std::cout << "\nBucket distribution:\n";
    for (size_t i = 0; i < hash_set.bucket_count(); ++i) {
        size_t bucket_size = hash_set.bucket_size(i);
        if (bucket_size > 0) {
            std::cout << "Bucket " << i << " has " << bucket_size << " elements\n";
        }
    }

    // ========== UNORDERED_MULTISET ==========
    separator("UNORDERED_MULTISET");

    // 6. Unordered Multiset
    std::cout << "\n6. UNORDERED_MULTISET (DUPLICATES ALLOWED):\n";
    std::unordered_multiset<int> umset = {5, 2, 8, 2, 9, 5, 1, 5};

    std::cout << "Elements: ";
    for (const auto& n : umset) {
        std::cout << n << " ";
    }
    std::cout << "\n";

    std::cout << "count(5) = " << umset.count(5) << "\n";
    std::cout << "count(2) = " << umset.count(2) << "\n";

    umset.insert(5);
    std::cout << "After insert(5), count(5) = " << umset.count(5) << "\n";

    // ========== UNORDERED_MAP ==========
    separator("UNORDERED_MAP");

    // 7. Map Initialization
    std::cout << "\n7. UNORDERED_MAP INITIALIZATION:\n";
    std::unordered_map<std::string, int> ages;
    std::unordered_map<std::string, int> scores = {
        {"Alice", 95},
        {"Bob", 87},
        {"Charlie", 92}
    };

    std::cout << "Scores (no guaranteed order):\n";
    for (const auto& [name, score] : scores) {
        std::cout << name << ": " << score << "\n";
    }

    // 8. Inserting into Map
    std::cout << "\n8. INSERTING INTO MAP:\n";
    std::unordered_map<int, std::string> students;

    students.insert({1, "John"});
    students.insert(std::make_pair(2, "Jane"));
    students[3] = "Jack";
    students.emplace(4, "Jill");

    std::cout << "Students:\n";
    for (const auto& [id, name] : students) {
        std::cout << id << ": " << name << "\n";
    }

    // 9. Accessing Map
    std::cout << "\n9. ACCESSING MAP:\n";
    std::unordered_map<std::string, double> prices = {
        {"apple", 1.20},
        {"banana", 0.80},
        {"orange", 1.50}
    };

    std::cout << "apple price: $" << prices["apple"] << "\n";
    std::cout << "banana price (at): $" << prices.at("banana") << "\n";

    // Safe access
    if (prices.contains("grape")) {
        std::cout << "grape price: $" << prices["grape"] << "\n";
    } else {
        std::cout << "grape not found\n";
    }

    // 10. Finding in Map
    std::cout << "\n10. FINDING IN MAP:\n";
    auto find_price = prices.find("orange");
    if (find_price != prices.end()) {
        std::cout << "Found: " << find_price->first << " = $" << find_price->second << "\n";
    }

    std::cout << "count('apple') = " << prices.count("apple") << "\n";
    std::cout << "contains('kiwi') = " << (prices.contains("kiwi") ? "true" : "false") << "\n";

    // 11. Modifying Map
    std::cout << "\n11. MODIFYING MAP:\n";
    std::unordered_map<std::string, int> inventory = {
        {"apples", 10},
        {"bananas", 5}
    };

    inventory["apples"] += 5;       // Modify
    inventory["oranges"] = 15;       // Add new
    inventory.at("bananas") = 20;    // Modify with at()

    std::cout << "Inventory:\n";
    for (const auto& [item, count] : inventory) {
        std::cout << item << ": " << count << "\n";
    }

    // 12. Removing from Map
    std::cout << "\n12. REMOVING FROM MAP:\n";
    std::unordered_map<std::string, int> data = {
        {"one", 1},
        {"two", 2},
        {"three", 3},
        {"four", 4}
    };

    data.erase("two");  // Erase by key
    std::cout << "After erase('two'), size = " << data.size() << "\n";

    // 13. Hash Function Info
    std::cout << "\n13. HASH PROPERTIES:\n";
    std::unordered_map<std::string, int> hash_map = {
        {"a", 1}, {"b", 2}, {"c", 3}, {"d", 4}, {"e", 5}
    };

    std::cout << "size() = " << hash_map.size() << "\n";
    std::cout << "bucket_count() = " << hash_map.bucket_count() << "\n";
    std::cout << "load_factor() = " << hash_map.load_factor() << "\n";

    // ========== UNORDERED_MULTIMAP ==========
    separator("UNORDERED_MULTIMAP");

    // 14. Unordered Multimap
    std::cout << "\n14. UNORDERED_MULTIMAP:\n";
    std::unordered_multimap<std::string, int> grades = {
        {"Math", 90},
        {"English", 85},
        {"Math", 95},
        {"Math", 92}
    };

    std::cout << "All grades:\n";
    for (const auto& [subject, grade] : grades) {
        std::cout << subject << ": " << grade << "\n";
    }

    std::cout << "\ncount('Math') = " << grades.count("Math") << "\n";

    // Get all Math grades
    std::cout << "All Math grades: ";
    auto range = grades.equal_range("Math");
    for (auto it = range.first; it != range.second; ++it) {
        std::cout << it->second << " ";
    }
    std::cout << "\n";

    // 15. Custom Hash Function
    separator("CUSTOM HASH");

    std::cout << "\n15. CUSTOM HASH FUNCTION:\n";

    // Custom struct
    struct Point {
        int x, y;
        bool operator==(const Point& other) const {
            return x == other.x && y == other.y;
        }
    };

    // Custom hash function
    struct PointHash {
        size_t operator()(const Point& p) const {
            return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
        }
    };

    std::unordered_set<Point, PointHash> point_set;
    point_set.insert({1, 2});
    point_set.insert({3, 4});
    point_set.insert({1, 2});  // Duplicate

    std::cout << "Point set size: " << point_set.size() << "\n";
    std::cout << "Points:\n";
    for (const auto& p : point_set) {
        std::cout << "(" << p.x << ", " << p.y << ")\n";
    }

    // 16. Performance Comparison
    separator("ORDERED VS UNORDERED");

    std::cout << "\nORDERED (set, map):\n";
    std::cout << "+ Sorted iteration\n";
    std::cout << "+ Predictable O(log n) performance\n";
    std::cout << "+ Range queries efficient\n";
    std::cout << "- Slower for pure lookups\n";

    std::cout << "\nUNORDERED (unordered_set, unordered_map):\n";
    std::cout << "+ Faster O(1) average lookup/insert\n";
    std::cout << "+ Better for pure hash table use\n";
    std::cout << "- No ordering\n";
    std::cout << "- Worst case O(n) with collisions\n";
    std::cout << "- More memory overhead\n";

    // 17. Use Cases
    separator("USE CASES");

    std::cout << "\nUse UNORDERED_SET when:\n";
    std::cout << "- Fast lookup is critical\n";
    std::cout << "- Order doesn't matter\n";
    std::cout << "- Checking membership frequently\n";
    std::cout << "- Example: Seen items, cache\n";

    std::cout << "\nUse UNORDERED_MAP when:\n";
    std::cout << "- Fast key-value lookup needed\n";
    std::cout << "- Order doesn't matter\n";
    std::cout << "- Implementing caches, dictionaries\n";
    std::cout << "- Frequency counting\n";

    std::cout << "\nUse SET/MAP when:\n";
    std::cout << "- Need sorted iteration\n";
    std::cout << "- Range queries needed\n";
    std::cout << "- Want predictable performance\n";

    // 18. Practical Example: Word Frequency Counter
    separator("PRACTICAL EXAMPLE");

    std::cout << "\n18. WORD FREQUENCY COUNTER:\n";
    std::vector<std::string> text = {
        "the", "quick", "brown", "fox", "jumps", "over",
        "the", "lazy", "dog", "the", "fox", "is", "quick"
    };

    std::unordered_map<std::string, int> word_count;
    for (const auto& word : text) {
        word_count[word]++;
    }

    std::cout << "Word frequencies:\n";
    for (const auto& [word, count] : word_count) {
        std::cout << word << ": " << count << "\n";
    }

    std::cout << "\n=== END OF UNORDERED CONTAINERS ===\n";

    return 0;
}
