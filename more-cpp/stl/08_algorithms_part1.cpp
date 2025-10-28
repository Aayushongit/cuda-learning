/**
 * 08_algorithms_part1.cpp
 *
 * STL ALGORITHMS PART 1
 * - Non-modifying sequence operations
 * - Modifying sequence operations
 * - Sorting and related operations
 * - Binary search operations
 *
 * All algorithms work with iterators (container-independent)
 * Most are in <algorithm> header
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

template<typename T>
void print(const std::vector<T>& vec, const std::string& label = "") {
    if (!label.empty()) std::cout << label << ": ";
    for (const auto& elem : vec) std::cout << elem << " ";
    std::cout << "\n";
}

int main() {
    std::cout << "=== ALGORITHMS PART 1 ===\n";

    // ========== NON-MODIFYING OPERATIONS ==========
    separator("NON-MODIFYING OPERATIONS");

    // 1. std::for_each
    std::cout << "\n1. FOR_EACH:\n";
    std::vector<int> nums = {1, 2, 3, 4, 5};
    std::for_each(nums.begin(), nums.end(), [](int n) {
        std::cout << n * n << " ";
    });
    std::cout << "\n";

    // 2. std::count & std::count_if
    std::cout << "\n2. COUNT & COUNT_IF:\n";
    std::vector<int> data = {1, 2, 3, 2, 4, 2, 5};
    int count_2 = std::count(data.begin(), data.end(), 2);
    std::cout << "Count of 2: " << count_2 << "\n";

    int count_even = std::count_if(data.begin(), data.end(),
                                    [](int n) { return n % 2 == 0; });
    std::cout << "Count of evens: " << count_even << "\n";

    // 3. std::find & std::find_if
    std::cout << "\n3. FIND & FIND_IF:\n";
    auto it = std::find(data.begin(), data.end(), 4);
    if (it != data.end()) {
        std::cout << "Found 4 at position: " << (it - data.begin()) << "\n";
    }

    auto it2 = std::find_if(data.begin(), data.end(),
                            [](int n) { return n > 3; });
    if (it2 != data.end()) {
        std::cout << "First element > 3: " << *it2 << "\n";
    }

    // 4. std::all_of, std::any_of, std::none_of
    std::cout << "\n4. ALL_OF, ANY_OF, NONE_OF:\n";
    std::vector<int> check = {2, 4, 6, 8, 10};

    bool all_even = std::all_of(check.begin(), check.end(),
                                [](int n) { return n % 2 == 0; });
    std::cout << "All even: " << (all_even ? "yes" : "no") << "\n";

    bool any_gt_5 = std::any_of(check.begin(), check.end(),
                                [](int n) { return n > 5; });
    std::cout << "Any > 5: " << (any_gt_5 ? "yes" : "no") << "\n";

    bool none_negative = std::none_of(check.begin(), check.end(),
                                      [](int n) { return n < 0; });
    std::cout << "None negative: " << (none_negative ? "yes" : "no") << "\n";

    // 5. std::min_element & std::max_element
    std::cout << "\n5. MIN_ELEMENT & MAX_ELEMENT:\n";
    std::vector<int> values = {3, 7, 1, 9, 4, 2};
    auto min_it = std::min_element(values.begin(), values.end());
    auto max_it = std::max_element(values.begin(), values.end());

    std::cout << "Min: " << *min_it << "\n";
    std::cout << "Max: " << *max_it << "\n";

    auto [min_e, max_e] = std::minmax_element(values.begin(), values.end());
    std::cout << "Minmax: [" << *min_e << ", " << *max_e << "]\n";

    // ========== MODIFYING OPERATIONS ==========
    separator("MODIFYING OPERATIONS");

    // 6. std::copy & std::copy_if
    std::cout << "\n6. COPY & COPY_IF:\n";
    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> dest1(5);
    std::copy(source.begin(), source.end(), dest1.begin());
    print(dest1, "Copy");

    std::vector<int> dest2;
    std::copy_if(source.begin(), source.end(), std::back_inserter(dest2),
                 [](int n) { return n % 2 == 0; });
    print(dest2, "Copy evens");

    // 7. std::move
    std::cout << "\n7. MOVE:\n";
    std::vector<std::string> src_strings = {"hello", "world", "cpp"};
    std::vector<std::string> dst_strings(3);
    std::move(src_strings.begin(), src_strings.end(), dst_strings.begin());
    print(dst_strings, "Moved strings");

    // 8. std::transform
    std::cout << "\n8. TRANSFORM:\n";
    std::vector<int> original = {1, 2, 3, 4, 5};
    std::vector<int> squared(5);
    std::transform(original.begin(), original.end(), squared.begin(),
                   [](int n) { return n * n; });
    print(squared, "Squared");

    // Binary transform
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};
    std::vector<int> sums(3);
    std::transform(a.begin(), a.end(), b.begin(), sums.begin(),
                   [](int x, int y) { return x + y; });
    print(sums, "Sums");

    // 9. std::fill & std::fill_n
    std::cout << "\n9. FILL & FILL_N:\n";
    std::vector<int> fill_vec(5);
    std::fill(fill_vec.begin(), fill_vec.end(), 99);
    print(fill_vec, "Filled");

    std::vector<int> fill_n_vec(10);
    std::fill_n(fill_n_vec.begin(), 5, 42);
    print(fill_n_vec, "Fill_n");

    // 10. std::generate & std::generate_n
    std::cout << "\n10. GENERATE & GENERATE_N:\n";
    std::vector<int> gen_vec(5);
    int counter = 0;
    std::generate(gen_vec.begin(), gen_vec.end(), [&counter]() {
        return counter++;
    });
    print(gen_vec, "Generated");

    // 11. std::replace & std::replace_if
    std::cout << "\n11. REPLACE & REPLACE_IF:\n";
    std::vector<int> replace_vec = {1, 2, 3, 2, 4, 2, 5};
    std::replace(replace_vec.begin(), replace_vec.end(), 2, 99);
    print(replace_vec, "Replace 2 with 99");

    std::vector<int> replace_if_vec = {1, 2, 3, 4, 5, 6, 7, 8};
    std::replace_if(replace_if_vec.begin(), replace_if_vec.end(),
                    [](int n) { return n % 2 == 0; }, 0);
    print(replace_if_vec, "Replace evens with 0");

    // 12. std::remove & std::remove_if (erase-remove idiom)
    std::cout << "\n12. REMOVE & REMOVE_IF:\n";
    std::vector<int> remove_vec = {1, 2, 3, 2, 4, 2, 5};
    print(remove_vec, "Before remove");

    auto new_end = std::remove(remove_vec.begin(), remove_vec.end(), 2);
    remove_vec.erase(new_end, remove_vec.end());
    print(remove_vec, "After remove(2)");

    std::vector<int> remove_if_vec = {1, 2, 3, 4, 5, 6, 7, 8};
    auto new_end2 = std::remove_if(remove_if_vec.begin(), remove_if_vec.end(),
                                    [](int n) { return n % 2 == 0; });
    remove_if_vec.erase(new_end2, remove_if_vec.end());
    print(remove_if_vec, "After remove_if (evens)");

    // 13. std::unique
    std::cout << "\n13. UNIQUE:\n";
    std::vector<int> unique_vec = {1, 1, 2, 2, 2, 3, 3, 4, 5, 5};
    print(unique_vec, "Before unique");

    auto new_end3 = std::unique(unique_vec.begin(), unique_vec.end());
    unique_vec.erase(new_end3, unique_vec.end());
    print(unique_vec, "After unique");

    // 14. std::reverse
    std::cout << "\n14. REVERSE:\n";
    std::vector<int> rev_vec = {1, 2, 3, 4, 5};
    std::reverse(rev_vec.begin(), rev_vec.end());
    print(rev_vec, "Reversed");

    // 15. std::rotate
    std::cout << "\n15. ROTATE:\n";
    std::vector<int> rot_vec = {1, 2, 3, 4, 5};
    std::rotate(rot_vec.begin(), rot_vec.begin() + 2, rot_vec.end());
    print(rot_vec, "Rotated by 2");

    // ========== SORTING ==========
    separator("SORTING");

    // 16. std::sort
    std::cout << "\n16. SORT:\n";
    std::vector<int> sort_vec = {5, 2, 8, 1, 9, 3};
    std::sort(sort_vec.begin(), sort_vec.end());
    print(sort_vec, "Sorted ascending");

    std::sort(sort_vec.begin(), sort_vec.end(), std::greater<int>());
    print(sort_vec, "Sorted descending");

    // Custom comparator
    std::vector<std::string> words = {"apple", "fig", "banana", "date"};
    std::sort(words.begin(), words.end(), [](const std::string& a, const std::string& b) {
        return a.length() < b.length();
    });
    print(words, "Sorted by length");

    // 17. std::stable_sort
    std::cout << "\n17. STABLE_SORT:\n";
    std::vector<std::pair<int, char>> pairs = {{1, 'a'}, {2, 'b'}, {1, 'c'}, {2, 'd'}};
    std::stable_sort(pairs.begin(), pairs.end(),
                     [](auto& a, auto& b) { return a.first < b.first; });
    std::cout << "Stable sort (preserves relative order): ";
    for (const auto& p : pairs) {
        std::cout << "(" << p.first << "," << p.second << ") ";
    }
    std::cout << "\n";

    // 18. std::partial_sort
    std::cout << "\n18. PARTIAL_SORT:\n";
    std::vector<int> partial_vec = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    std::partial_sort(partial_vec.begin(), partial_vec.begin() + 3, partial_vec.end());
    print(partial_vec, "First 3 sorted");

    // 19. std::nth_element
    std::cout << "\n19. NTH_ELEMENT:\n";
    std::vector<int> nth_vec = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    std::nth_element(nth_vec.begin(), nth_vec.begin() + 4, nth_vec.end());
    std::cout << "Median (5th element): " << nth_vec[4] << "\n";
    print(nth_vec, "Partitioned around median");

    // 20. std::is_sorted
    std::cout << "\n20. IS_SORTED:\n";
    std::vector<int> sorted = {1, 2, 3, 4, 5};
    std::vector<int> unsorted = {1, 3, 2, 4, 5};
    std::cout << "sorted is_sorted: " << (std::is_sorted(sorted.begin(), sorted.end()) ? "yes" : "no") << "\n";
    std::cout << "unsorted is_sorted: " << (std::is_sorted(unsorted.begin(), unsorted.end()) ? "yes" : "no") << "\n";

    // ========== BINARY SEARCH ==========
    separator("BINARY SEARCH (on sorted ranges)");

    // 21. std::binary_search
    std::cout << "\n21. BINARY_SEARCH:\n";
    std::vector<int> sorted_vec = {1, 3, 5, 7, 9, 11, 13};
    bool found = std::binary_search(sorted_vec.begin(), sorted_vec.end(), 7);
    std::cout << "7 found: " << (found ? "yes" : "no") << "\n";

    // 22. std::lower_bound & std::upper_bound
    std::cout << "\n22. LOWER_BOUND & UPPER_BOUND:\n";
    auto lb = std::lower_bound(sorted_vec.begin(), sorted_vec.end(), 7);
    auto ub = std::upper_bound(sorted_vec.begin(), sorted_vec.end(), 7);
    std::cout << "lower_bound(7): " << *lb << " at index " << (lb - sorted_vec.begin()) << "\n";
    std::cout << "upper_bound(7): " << *ub << " at index " << (ub - sorted_vec.begin()) << "\n";

    // 23. std::equal_range
    std::cout << "\n23. EQUAL_RANGE:\n";
    std::vector<int> multi = {1, 2, 2, 2, 3, 4};
    auto [first, last] = std::equal_range(multi.begin(), multi.end(), 2);
    std::cout << "Range of 2: [" << (first - multi.begin()) << ", "
              << (last - multi.begin()) << ")\n";

    // ========== PARTITIONING ==========
    separator("PARTITIONING");

    // 24. std::partition
    std::cout << "\n24. PARTITION:\n";
    std::vector<int> part_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto partition_point = std::partition(part_vec.begin(), part_vec.end(),
                                          [](int n) { return n % 2 == 0; });
    print(part_vec, "Partitioned (evens first)");
    std::cout << "Partition point index: " << (partition_point - part_vec.begin()) << "\n";

    // 25. std::is_partitioned
    std::cout << "\n25. IS_PARTITIONED:\n";
    bool partitioned = std::is_partitioned(part_vec.begin(), part_vec.end(),
                                           [](int n) { return n % 2 == 0; });
    std::cout << "Is partitioned: " << (partitioned ? "yes" : "no") << "\n";

    std::cout << "\n=== END OF ALGORITHMS PART 1 ===\n";

    return 0;
}
