/**
 * 07_iterators.cpp
 *
 * ITERATORS - Objects that point to elements in containers
 *
 * Iterator Categories (in order of capability):
 * 1. Input Iterator: Read-only, forward only, single-pass
 * 2. Output Iterator: Write-only, forward only, single-pass
 * 3. Forward Iterator: Read/write, forward only, multi-pass
 * 4. Bidirectional Iterator: Forward + backward movement
 * 5. Random Access Iterator: Jump to any position, arithmetic
 * 6. Contiguous Iterator (C++20): Random access + contiguous memory
 *
 * Special Iterators:
 * - Reverse iterators
 * - Insert iterators
 * - Stream iterators
 * - Move iterators
 */

#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <iterator>
#include <algorithm>
#include <sstream>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== ITERATORS ===\n";

    // ========== BASIC ITERATORS ==========
    separator("BASIC ITERATORS");

    // 1. Iterator Basics
    std::cout << "\n1. ITERATOR BASICS:\n";
    std::vector<int> vec = {10, 20, 30, 40, 50};

    // begin() and end()
    std::vector<int>::iterator it = vec.begin();
    std::cout << "First element (*begin()): " << *it << "\n";

    std::vector<int>::iterator end_it = vec.end();  // Points PAST last element
    --end_it;  // Now points to last element
    std::cout << "Last element (*(end-1)): " << *end_it << "\n";

    // Iteration
    std::cout << "Forward iteration: ";
    for (auto iter = vec.begin(); iter != vec.end(); ++iter) {
        std::cout << *iter << " ";
    }
    std::cout << "\n";

    // 2. Const Iterators
    std::cout << "\n2. CONST ITERATORS:\n";
    const std::vector<int> const_vec = {1, 2, 3, 4, 5};

    // const_iterator prevents modification
    std::vector<int>::const_iterator cit = const_vec.begin();
    std::cout << "Value: " << *cit << "\n";
    // *cit = 100;  // ERROR: Cannot modify through const_iterator

    // cbegin() and cend() (always const)
    std::cout << "Using cbegin/cend: ";
    for (auto it = vec.cbegin(); it != vec.cend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";

    // 3. Reverse Iterators
    std::cout << "\n3. REVERSE ITERATORS:\n";
    std::vector<int> rvec = {1, 2, 3, 4, 5};

    std::cout << "Reverse iteration: ";
    for (auto rit = rvec.rbegin(); rit != rvec.rend(); ++rit) {
        std::cout << *rit << " ";
    }
    std::cout << "\n";

    std::cout << "Const reverse iteration: ";
    for (auto crit = rvec.crbegin(); crit != rvec.crend(); ++crit) {
        std::cout << *crit << " ";
    }
    std::cout << "\n";

    // ========== ITERATOR OPERATIONS ==========
    separator("ITERATOR OPERATIONS");

    // 4. Iterator Arithmetic (Random Access)
    std::cout << "\n4. ITERATOR ARITHMETIC (Vector):\n";
    std::vector<int> nums = {0, 10, 20, 30, 40, 50};

    auto it1 = nums.begin();
    auto it2 = it1 + 3;  // Jump 3 positions
    std::cout << "begin() + 3 = " << *it2 << "\n";

    auto it3 = nums.end() - 2;  // 2 before end
    std::cout << "end() - 2 = " << *it3 << "\n";

    // Difference between iterators
    auto diff = it3 - it1;
    std::cout << "Distance: " << diff << "\n";

    // Comparison
    std::cout << "it1 < it3: " << (it1 < it3 ? "true" : "false") << "\n";

    // 5. std::advance, std::distance, std::next, std::prev
    std::cout << "\n5. ITERATOR HELPER FUNCTIONS:\n";
    std::list<int> lst = {100, 200, 300, 400, 500};

    auto list_it = lst.begin();
    std::cout << "Initial: " << *list_it << "\n";

    std::advance(list_it, 2);  // Move 2 positions
    std::cout << "After advance(2): " << *list_it << "\n";

    auto dist = std::distance(lst.begin(), list_it);
    std::cout << "Distance from begin: " << dist << "\n";

    // next() - returns new iterator
    auto next_it = std::next(list_it);
    std::cout << "next(): " << *next_it << "\n";
    std::cout << "Original unchanged: " << *list_it << "\n";

    // next with offset
    auto next_it2 = std::next(lst.begin(), 3);
    std::cout << "next(begin(), 3): " << *next_it2 << "\n";

    // prev() - returns new iterator
    auto prev_it = std::prev(lst.end());
    std::cout << "prev(end()): " << *prev_it << "\n";

    // ========== INSERT ITERATORS ==========
    separator("INSERT ITERATORS");

    // 6. Back Insert Iterator
    std::cout << "\n6. BACK INSERT ITERATOR:\n";
    std::vector<int> dest1;
    auto back_it = std::back_inserter(dest1);

    *back_it = 10;  // Calls push_back(10)
    *back_it = 20;
    *back_it = 30;

    std::cout << "dest1: ";
    for (int n : dest1) std::cout << n << " ";
    std::cout << "\n";

    // With copy algorithm
    std::vector<int> source = {100, 200, 300};
    std::copy(source.begin(), source.end(), std::back_inserter(dest1));

    std::cout << "After copy: ";
    for (int n : dest1) std::cout << n << " ";
    std::cout << "\n";

    // 7. Front Insert Iterator
    std::cout << "\n7. FRONT INSERT ITERATOR:\n";
    std::list<int> dest2;  // deque and list support push_front

    auto front_it = std::front_inserter(dest2);
    *front_it = 10;
    *front_it = 20;
    *front_it = 30;

    std::cout << "dest2 (LIFO order): ";
    for (int n : dest2) std::cout << n << " ";
    std::cout << "\n";

    // 8. General Insert Iterator
    std::cout << "\n8. INSERT ITERATOR:\n";
    std::vector<int> dest3 = {1, 2, 3, 7, 8, 9};

    // Insert at specific position
    auto insert_pos = dest3.begin() + 3;
    auto insert_it = std::inserter(dest3, insert_pos);

    *insert_it = 4;
    *insert_it = 5;
    *insert_it = 6;

    std::cout << "dest3: ";
    for (int n : dest3) std::cout << n << " ";
    std::cout << "\n";

    // ========== STREAM ITERATORS ==========
    separator("STREAM ITERATORS");

    // 9. istream_iterator
    std::cout << "\n9. ISTREAM_ITERATOR:\n";
    std::istringstream iss("10 20 30 40 50");

    std::istream_iterator<int> is_it(iss);
    std::istream_iterator<int> is_end;  // End-of-stream iterator

    std::vector<int> from_stream;
    while (is_it != is_end) {
        from_stream.push_back(*is_it);
        ++is_it;
    }

    std::cout << "Read from stream: ";
    for (int n : from_stream) std::cout << n << " ";
    std::cout << "\n";

    // Direct construction - Using braces to avoid most vexing parse
    std::istringstream iss2("100 200 300");
    std::vector<int> direct{
        std::istream_iterator<int>(iss2),
        std::istream_iterator<int>()
    };

    std::cout << "Direct construction: ";
    for (int n : direct) std::cout << n << " ";
    std::cout << "\n";

    // 10. ostream_iterator
    std::cout << "\n10. OSTREAM_ITERATOR:\n";
    std::vector<int> output = {1, 2, 3, 4, 5};

    std::cout << "Output with space: ";
    std::ostream_iterator<int> os_it(std::cout, " ");
    std::copy(output.begin(), output.end(), os_it);
    std::cout << "\n";

    std::cout << "Output with comma: ";
    std::ostream_iterator<int> os_it2(std::cout, ", ");
    for (int n : output) {
        *os_it2 = n;
        ++os_it2;
    }
    std::cout << "\n";

    // ========== MOVE ITERATORS ==========
    separator("MOVE ITERATORS");

    // 11. move_iterator
    std::cout << "\n11. MOVE ITERATOR:\n";
    std::vector<std::string> source_strings = {"hello", "world", "cpp", "stl"};

    std::cout << "Source before move: ";
    for (const auto& s : source_strings) std::cout << s << " ";
    std::cout << "\n";

    std::vector<std::string> dest_strings;
    std::move(
        std::make_move_iterator(source_strings.begin()),
        std::make_move_iterator(source_strings.end()),
        std::back_inserter(dest_strings)
    );

    std::cout << "Destination after move: ";
    for (const auto& s : dest_strings) std::cout << s << " ";
    std::cout << "\n";

    std::cout << "Source after move (moved-from): ";
    for (const auto& s : source_strings) std::cout << "[" << s << "] ";
    std::cout << "\n";

    // ========== ITERATOR CATEGORIES ==========
    separator("ITERATOR CATEGORIES");

    // 12. Iterator Category Examples
    std::cout << "\n12. ITERATOR CATEGORIES:\n";

    std::cout << "\nRANDOM ACCESS (vector, deque, array):\n";
    std::cout << "- it + n, it - n\n";
    std::cout << "- it[n]\n";
    std::cout << "- it1 < it2, it1 > it2\n";

    std::cout << "\nBIDIRECTIONAL (list, set, map):\n";
    std::cout << "- ++it, --it\n";
    std::cout << "- Cannot: it + n, it[n]\n";

    std::cout << "\nFORWARD (forward_list, unordered containers):\n";
    std::cout << "- ++it only\n";
    std::cout << "- Cannot: --it\n";

    // 13. Iterator Traits
    std::cout << "\n13. ITERATOR TRAITS:\n";

    using vec_iterator = std::vector<int>::iterator;
    using list_iterator = std::list<int>::iterator;

    std::cout << "Vector iterator category: ";
    if (std::is_same_v<
        typename std::iterator_traits<vec_iterator>::iterator_category,
        std::random_access_iterator_tag>) {
        std::cout << "Random Access\n";
    }

    std::cout << "List iterator category: ";
    if (std::is_same_v<
        typename std::iterator_traits<list_iterator>::iterator_category,
        std::bidirectional_iterator_tag>) {
        std::cout << "Bidirectional\n";
    }

    // ========== PRACTICAL EXAMPLES ==========
    separator("PRACTICAL EXAMPLES");

    // 14. Finding Elements
    std::cout << "\n14. FINDING WITH ITERATORS:\n";
    std::vector<int> search_vec = {1, 3, 5, 7, 9, 11, 13};

    auto found = std::find(search_vec.begin(), search_vec.end(), 7);
    if (found != search_vec.end()) {
        std::cout << "Found 7 at position: " << (found - search_vec.begin()) << "\n";
    }

    auto found_if = std::find_if(search_vec.begin(), search_vec.end(),
                                  [](int n) { return n > 10; });
    if (found_if != search_vec.end()) {
        std::cout << "First element > 10: " << *found_if << "\n";
    }

    // 15. Range Modification
    std::cout << "\n15. RANGE MODIFICATION:\n";
    std::vector<int> modify_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::cout << "Original: ";
    for (int n : modify_vec) std::cout << n << " ";
    std::cout << "\n";

    // Double middle elements
    auto mid_begin = modify_vec.begin() + 3;
    auto mid_end = modify_vec.begin() + 7;
    std::transform(mid_begin, mid_end, mid_begin, [](int n) { return n * 2; });

    std::cout << "After doubling [3:7): ";
    for (int n : modify_vec) std::cout << n << " ";
    std::cout << "\n";

    // 16. Erasing with Iterators
    std::cout << "\n16. ERASING WITH ITERATORS:\n";
    std::vector<int> erase_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    std::cout << "Original: ";
    for (int n : erase_vec) std::cout << n << " ";
    std::cout << "\n";

    // Erase even numbers (erase-remove idiom)
    auto new_end = std::remove_if(erase_vec.begin(), erase_vec.end(),
                                   [](int n) { return n % 2 == 0; });
    erase_vec.erase(new_end, erase_vec.end());

    std::cout << "After removing evens: ";
    for (int n : erase_vec) std::cout << n << " ";
    std::cout << "\n";

    // ========== BEST PRACTICES ==========
    separator("BEST PRACTICES");

    std::cout << "\n1. Use auto for iterator types\n";
    std::cout << "2. Prefer range-based for when possible\n";
    std::cout << "3. Be careful with iterator invalidation:\n";
    std::cout << "   - vector: insert/erase/resize invalidate\n";
    std::cout << "   - list: only erased elements invalidated\n";
    std::cout << "4. Use const_iterator when not modifying\n";
    std::cout << "5. Check iterator != end() before dereferencing\n";
    std::cout << "6. Use std::next/prev instead of arithmetic for portability\n";

    std::cout << "\n=== END OF ITERATORS ===\n";

    return 0;
}
