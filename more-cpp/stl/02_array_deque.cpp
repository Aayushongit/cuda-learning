/**
 * 02_array_deque.cpp
 *
 * ARRAY - Fixed-size container (compile-time size)
 * DEQUE - Double-ended queue (efficient insertion at both ends)
 *
 * ARRAY Features:
 * - Fixed size at compile time
 * - No dynamic memory allocation
 * - Random access O(1)
 * - Stack allocated (fast)
 *
 * DEQUE Features:
 * - Random access O(1)
 * - Fast insertion/deletion at both ends O(1)
 * - Not contiguous in memory
 */

#include <iostream>
#include <array>
#include <deque>
#include <algorithm>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== ARRAY AND DEQUE ===\n";

    // ========== STD::ARRAY ==========
    separator("STD::ARRAY");

    // 1. Declaration and Initialization
    std::cout << "\n1. ARRAY INITIALIZATION:\n";
    std::array<int, 5> arr1;                    // Uninitialized
    std::array<int, 5> arr2 = {1, 2, 3, 4, 5}; // Initialize with values
    std::array<int, 5> arr3 = {1, 2};          // Partial init, rest are 0
    std::array<int, 5> arr4{};                 // All elements zero-initialized

    std::cout << "arr2: ";
    for (const auto& val : arr2) std::cout << val << " ";
    std::cout << "\n";

    std::cout << "arr3 (partial): ";
    for (const auto& val : arr3) std::cout << val << " ";
    std::cout << "\n";

    // 2. Accessing Elements
    std::cout << "\n2. ARRAY ACCESS:\n";
    std::array<int, 5> arr = {10, 20, 30, 40, 50};

    std::cout << "arr[0] = " << arr[0] << "\n";
    std::cout << "arr.at(1) = " << arr.at(1) << "\n";  // Bounds checking
    std::cout << "arr.front() = " << arr.front() << "\n";
    std::cout << "arr.back() = " << arr.back() << "\n";

    // Get raw pointer
    int* data = arr.data();
    std::cout << "Via data(): " << data[2] << "\n";

    // 3. Array Properties
    std::cout << "\n3. ARRAY PROPERTIES:\n";
    std::cout << "size() = " << arr.size() << "\n";
    std::cout << "max_size() = " << arr.max_size() << "\n";
    std::cout << "empty() = " << (arr.empty() ? "true" : "false") << "\n";

    // 4. Modifying Array
    std::cout << "\n4. MODIFYING ARRAY:\n";
    std::array<int, 5> modify = {1, 2, 3, 4, 5};
    std::cout << "Original: ";
    for (const auto& v : modify) std::cout << v << " ";
    std::cout << "\n";

    modify.fill(99);  // Fill all elements with 99
    std::cout << "After fill(99): ";
    for (const auto& v : modify) std::cout << v << " ";
    std::cout << "\n";

    // 5. Swapping Arrays
    std::cout << "\n5. SWAPPING ARRAYS:\n";
    std::array<int, 3> a1 = {1, 2, 3};
    std::array<int, 3> a2 = {7, 8, 9};

    std::cout << "Before swap - a1: ";
    for (const auto& v : a1) std::cout << v << " ";
    std::cout << "\n";

    a1.swap(a2);

    std::cout << "After swap - a1: ";
    for (const auto& v : a1) std::cout << v << " ";
    std::cout << "\n";

    // 6. Iterators with Array
    std::cout << "\n6. ARRAY ITERATORS:\n";
    std::array<int, 5> iter_arr = {5, 3, 8, 1, 9};

    std::cout << "Forward: ";
    for (auto it = iter_arr.begin(); it != iter_arr.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";

    std::cout << "Reverse: ";
    for (auto it = iter_arr.rbegin(); it != iter_arr.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";

    // 7. Algorithms with Array
    std::cout << "\n7. ALGORITHMS WITH ARRAY:\n";
    std::array<int, 6> algo_arr = {3, 1, 4, 1, 5, 9};

    std::cout << "Original: ";
    for (const auto& v : algo_arr) std::cout << v << " ";
    std::cout << "\n";

    std::sort(algo_arr.begin(), algo_arr.end());
    std::cout << "Sorted: ";
    for (const auto& v : algo_arr) std::cout << v << " ";
    std::cout << "\n";

    // 8. Multidimensional Array
    std::cout << "\n8. MULTIDIMENSIONAL ARRAY:\n";
    std::array<std::array<int, 3>, 2> matrix = {{
        {1, 2, 3},
        {4, 5, 6}
    }};

    std::cout << "Matrix:\n";
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    // 9. Structured Bindings (C++17)
    std::cout << "\n9. STRUCTURED BINDINGS:\n";
    std::array<int, 3> coords = {10, 20, 30};
    auto [x, y, z] = coords;
    std::cout << "x=" << x << ", y=" << y << ", z=" << z << "\n";

    // ========== STD::DEQUE ==========
    separator("STD::DEQUE");

    // 10. Deque Initialization
    std::cout << "\n10. DEQUE INITIALIZATION:\n";
    std::deque<int> deq1;                      // Empty
    std::deque<int> deq2(5);                   // 5 elements, default 0
    std::deque<int> deq3(5, 100);              // 5 elements, value 100
    std::deque<int> deq4 = {1, 2, 3, 4, 5};   // Initializer list

    std::cout << "deq4: ";
    for (const auto& v : deq4) std::cout << v << " ";
    std::cout << "\n";

    // 11. Adding Elements to Deque
    std::cout << "\n11. ADDING TO DEQUE:\n";
    std::deque<int> deq;

    // Add at back
    deq.push_back(10);
    deq.push_back(20);
    deq.push_back(30);

    // Add at front (efficient in deque!)
    deq.push_front(5);
    deq.push_front(1);

    std::cout << "After push operations: ";
    for (const auto& v : deq) std::cout << v << " ";
    std::cout << "\n";

    // Emplace (construct in-place)
    deq.emplace_back(40);
    deq.emplace_front(0);

    std::cout << "After emplace operations: ";
    for (const auto& v : deq) std::cout << v << " ";
    std::cout << "\n";

    // Insert at position
    deq.insert(deq.begin() + 3, 99);
    std::cout << "After insert at position 3: ";
    for (const auto& v : deq) std::cout << v << " ";
    std::cout << "\n";

    // 12. Accessing Deque Elements
    std::cout << "\n12. ACCESSING DEQUE:\n";
    std::deque<int> access_deq = {10, 20, 30, 40, 50};

    std::cout << "deq[0] = " << access_deq[0] << "\n";
    std::cout << "deq.at(2) = " << access_deq.at(2) << "\n";
    std::cout << "deq.front() = " << access_deq.front() << "\n";
    std::cout << "deq.back() = " << access_deq.back() << "\n";

    // 13. Removing from Deque
    std::cout << "\n13. REMOVING FROM DEQUE:\n";
    std::deque<int> remove_deq = {1, 2, 3, 4, 5, 6, 7, 8};

    std::cout << "Original: ";
    for (const auto& v : remove_deq) std::cout << v << " ";
    std::cout << "\n";

    remove_deq.pop_front();  // Remove from front (efficient!)
    std::cout << "After pop_front(): ";
    for (const auto& v : remove_deq) std::cout << v << " ";
    std::cout << "\n";

    remove_deq.pop_back();   // Remove from back
    std::cout << "After pop_back(): ";
    for (const auto& v : remove_deq) std::cout << v << " ";
    std::cout << "\n";

    remove_deq.erase(remove_deq.begin() + 2);  // Erase at position
    std::cout << "After erase at position 2: ";
    for (const auto& v : remove_deq) std::cout << v << " ";
    std::cout << "\n";

    // 14. Deque Size and Capacity
    std::cout << "\n14. DEQUE SIZE:\n";
    std::deque<int> size_deq = {1, 2, 3, 4, 5};
    std::cout << "size() = " << size_deq.size() << "\n";
    std::cout << "max_size() = " << size_deq.max_size() << "\n";
    std::cout << "empty() = " << (size_deq.empty() ? "true" : "false") << "\n";

    size_deq.resize(8, 99);  // Resize to 8, new elements = 99
    std::cout << "After resize(8, 99): ";
    for (const auto& v : size_deq) std::cout << v << " ";
    std::cout << "\n";

    size_deq.shrink_to_fit();  // Reduce memory
    std::cout << "After shrink_to_fit()\n";

    // 15. Deque Iterators
    std::cout << "\n15. DEQUE ITERATORS:\n";
    std::deque<int> iter_deq = {10, 20, 30, 40, 50};

    std::cout << "Forward iteration: ";
    for (auto it = iter_deq.begin(); it != iter_deq.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";

    std::cout << "Reverse iteration: ";
    for (auto it = iter_deq.rbegin(); it != iter_deq.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";

    // 16. Deque vs Vector
    separator("DEQUE VS VECTOR");

    std::cout << "\nDEQUE Advantages:\n";
    std::cout << "- Fast insertion/deletion at BOTH ends O(1)\n";
    std::cout << "- No reallocation when growing\n";
    std::cout << "- Iterators less likely to be invalidated\n";

    std::cout << "\nVECTOR Advantages:\n";
    std::cout << "- Contiguous memory (better cache locality)\n";
    std::cout << "- Slightly faster random access\n";
    std::cout << "- Less memory overhead\n";

    // 17. Use Cases
    separator("USE CASES");

    std::cout << "\nUse ARRAY when:\n";
    std::cout << "- Size is known at compile time\n";
    std::cout << "- Need stack allocation\n";
    std::cout << "- Replace C-style arrays\n";

    std::cout << "\nUse DEQUE when:\n";
    std::cout << "- Need to add/remove from both ends\n";
    std::cout << "- Implementing queues\n";
    std::cout << "- Don't need contiguous memory\n";

    std::cout << "\n=== END OF ARRAY AND DEQUE ===\n";

    return 0;
}
