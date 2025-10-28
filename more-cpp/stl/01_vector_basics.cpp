/**
 * 01_vector_basics.cpp
 *
 * VECTOR - Dynamic array that can grow/shrink
 * Most commonly used STL container
 *
 * Key Features:
 * - Random access in O(1)
 * - Fast insertion/deletion at end O(1) amortized
 * - Slow insertion/deletion in middle O(n)
 * - Contiguous memory storage
 */

#include <iostream>
#include <vector>
#include <algorithm>

void printVector(const std::vector<int>& vec, const std::string& label) {
    std::cout << label << ": ";
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=== VECTOR BASICS ===\n\n";

    // 1. Declaration and Initialization
    std::cout << "1. INITIALIZATION METHODS:\n";
    std::vector<int> vec1;                          // Empty vector
    std::vector<int> vec2(5);                       // 5 elements, default initialized to 0
    std::vector<int> vec3(5, 100);                  // 5 elements, all initialized to 100
    std::vector<int> vec4 = {1, 2, 3, 4, 5};       // Initialize with list
    std::vector<int> vec5(vec4);                    // Copy constructor
    std::vector<int> vec6(vec4.begin(), vec4.end()); // Range constructor

    printVector(vec2, "vec2 (5 elements, default)");
    printVector(vec3, "vec3 (5 elements, value 100)");
    printVector(vec4, "vec4 (initializer list)");

    // 2. Adding Elements
    std::cout << "\n2. ADDING ELEMENTS:\n";
    std::vector<int> numbers;
    numbers.push_back(10);      // Add at end
    numbers.push_back(20);
    numbers.push_back(30);
    numbers.emplace_back(40);   // Construct in-place (more efficient)

    printVector(numbers, "After push_back/emplace_back");

    numbers.insert(numbers.begin() + 1, 15);  // Insert at position 1
    printVector(numbers, "After insert at position 1");

    numbers.insert(numbers.end(), {50, 60, 70});  // Insert multiple
    printVector(numbers, "After inserting multiple at end");

    // 3. Accessing Elements
    std::cout << "\n3. ACCESSING ELEMENTS:\n";
    std::cout << "numbers[0] = " << numbers[0] << "\n";           // No bounds checking
    std::cout << "numbers.at(1) = " << numbers.at(1) << "\n";     // With bounds checking
    std::cout << "numbers.front() = " << numbers.front() << "\n";  // First element
    std::cout << "numbers.back() = " << numbers.back() << "\n";    // Last element

    // Direct data pointer access
    int* ptr = numbers.data();
    std::cout << "Via data pointer: " << ptr[0] << "\n";

    // 4. Size and Capacity
    std::cout << "\n4. SIZE AND CAPACITY:\n";
    std::cout << "size() = " << numbers.size() << "\n";           // Number of elements
    std::cout << "capacity() = " << numbers.capacity() << "\n";   // Allocated space
    std::cout << "max_size() = " << numbers.max_size() << "\n";   // Theoretical maximum
    std::cout << "empty() = " << (numbers.empty() ? "true" : "false") << "\n";

    // Reserve space to avoid reallocations
    numbers.reserve(100);
    std::cout << "After reserve(100), capacity = " << numbers.capacity() << "\n";

    // Shrink capacity to fit size
    numbers.shrink_to_fit();
    std::cout << "After shrink_to_fit(), capacity = " << numbers.capacity() << "\n";

    // 5. Removing Elements
    std::cout << "\n5. REMOVING ELEMENTS:\n";
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    printVector(data, "Original");

    data.pop_back();  // Remove last element
    printVector(data, "After pop_back()");

    data.erase(data.begin() + 2);  // Remove element at index 2
    printVector(data, "After erase(index 2)");

    data.erase(data.begin() + 1, data.begin() + 4);  // Remove range
    printVector(data, "After erase(range 1-4)");

    data.clear();  // Remove all elements
    std::cout << "After clear(), size = " << data.size() << "\n";

    // 6. Modifying Elements
    std::cout << "\n6. MODIFYING ELEMENTS:\n";
    std::vector<int> modify = {1, 2, 3, 4, 5};
    printVector(modify, "Original");

    modify[2] = 99;  // Direct assignment
    printVector(modify, "After modify[2] = 99");

    modify.assign(5, 10);  // Assign 5 elements with value 10
    printVector(modify, "After assign(5, 10)");

    modify.assign({1, 2, 3});  // Assign from initializer list
    printVector(modify, "After assign from list");

    // 7. Iterating
    std::cout << "\n7. ITERATION METHODS:\n";
    std::vector<int> iter_vec = {10, 20, 30, 40, 50};

    std::cout << "Range-based for loop: ";
    for (const auto& val : iter_vec) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    std::cout << "Iterator: ";
    for (auto it = iter_vec.begin(); it != iter_vec.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";

    std::cout << "Reverse iterator: ";
    for (auto it = iter_vec.rbegin(); it != iter_vec.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";

    // 8. Algorithms with Vectors
    std::cout << "\n8. COMMON ALGORITHMS:\n";
    std::vector<int> algo_vec = {5, 2, 8, 1, 9, 3, 7};
    printVector(algo_vec, "Original");

    std::sort(algo_vec.begin(), algo_vec.end());
    printVector(algo_vec, "After sort");

    std::reverse(algo_vec.begin(), algo_vec.end());
    printVector(algo_vec, "After reverse");

    auto it = std::find(algo_vec.begin(), algo_vec.end(), 8);
    if (it != algo_vec.end()) {
        std::cout << "Found 8 at position: " << (it - algo_vec.begin()) << "\n";
    }

    // 9. 2D Vectors
    std::cout << "\n9. 2D VECTORS (MATRIX):\n";
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::cout << "Matrix:\n";
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    // Dynamic 2D vector
    int rows = 3, cols = 4;
    std::vector<std::vector<int>> dynamic_matrix(rows, std::vector<int>(cols, 0));
    std::cout << "\nDynamic 3x4 matrix initialized to 0\n";

    // 10. Vector of Custom Objects
    std::cout << "\n10. VECTOR OF CUSTOM OBJECTS:\n";
    struct Point {
        int x, y;
        Point(int x, int y) : x(x), y(y) {}
    };

    std::vector<Point> points;
    points.push_back(Point(1, 2));
    points.emplace_back(3, 4);  // More efficient, constructs in-place
    points.emplace_back(5, 6);

    std::cout << "Points: ";
    for (const auto& p : points) {
        std::cout << "(" << p.x << "," << p.y << ") ";
    }
    std::cout << "\n";

    // 11. Performance Tips
    std::cout << "\n11. PERFORMANCE TIPS:\n";
    std::vector<int> perf_vec;

    // Good: Reserve space if you know size
    perf_vec.reserve(1000);
    std::cout << "Reserved space for 1000 elements\n";

    // Good: Use emplace_back instead of push_back for complex objects
    std::cout << "Use emplace_back for in-place construction\n";

    // Good: Pass vectors by const reference to avoid copying
    std::cout << "Pass vectors as const& to functions\n";

    std::cout << "\n=== END OF VECTOR BASICS ===\n";

    return 0;
}
