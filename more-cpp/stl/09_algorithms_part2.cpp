/**
 * 09_algorithms_part2.cpp
 *
 * STL ALGORITHMS PART 2
 * - Numeric algorithms
 * - Heap operations
 * - Set operations
 * - Permutation operations
 * - Other useful algorithms
 */

#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <functional>
#include <random>

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
    std::cout << "=== ALGORITHMS PART 2 ===\n";

    // ========== NUMERIC ALGORITHMS ==========
    separator("NUMERIC ALGORITHMS");

    // 1. std::accumulate
    std::cout << "\n1. ACCUMULATE:\n";
    std::vector<int> nums = {1, 2, 3, 4, 5};
    int sum = std::accumulate(nums.begin(), nums.end(), 0);
    std::cout << "Sum: " << sum << "\n";

    int product = std::accumulate(nums.begin(), nums.end(), 1,
                                   std::multiplies<int>());
    std::cout << "Product: " << product << "\n";

    // 2. std::reduce (C++17, parallel-friendly)
    std::cout << "\n2. REDUCE:\n";
    std::vector<int> vals = {10, 20, 30, 40, 50};
    int total = std::reduce(vals.begin(), vals.end());
    std::cout << "Reduce sum: " << total << "\n";

    // 3. std::inner_product
    std::cout << "\n3. INNER_PRODUCT:\n";
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};
    int dot_product = std::inner_product(a.begin(), a.end(), b.begin(), 0);
    std::cout << "Dot product: " << dot_product << "\n";  // 1*4 + 2*5 + 3*6 = 32

    // 4. std::partial_sum
    std::cout << "\n4. PARTIAL_SUM:\n";
    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> cumulative(5);
    std::partial_sum(source.begin(), source.end(), cumulative.begin());
    print(cumulative, "Cumulative sum");

    // 5. std::adjacent_difference
    std::cout << "\n5. ADJACENT_DIFFERENCE:\n";
    std::vector<int> sequence = {10, 20, 35, 55, 80};
    std::vector<int> differences(5);
    std::adjacent_difference(sequence.begin(), sequence.end(), differences.begin());
    print(differences, "Adjacent differences");

    // 6. std::iota
    std::cout << "\n6. IOTA:\n";
    std::vector<int> iota_vec(10);
    std::iota(iota_vec.begin(), iota_vec.end(), 1);  // Start from 1
    print(iota_vec, "Iota from 1");

    // 7. std::transform_reduce (C++17)
    std::cout << "\n7. TRANSFORM_REDUCE:\n";
    std::vector<int> x = {1, 2, 3, 4};
    std::vector<int> y = {1, 2, 3, 4};
    // Sum of squares
    int sum_of_squares = std::transform_reduce(
        x.begin(), x.end(), y.begin(), 0);
    std::cout << "Sum of element-wise products: " << sum_of_squares << "\n";

    // ========== HEAP OPERATIONS ==========
    separator("HEAP OPERATIONS");

    // 8. std::make_heap
    std::cout << "\n8. MAKE_HEAP:\n";
    std::vector<int> heap_vec = {3, 1, 4, 1, 5, 9, 2, 6};
    print(heap_vec, "Before make_heap");
    std::make_heap(heap_vec.begin(), heap_vec.end());
    print(heap_vec, "After make_heap (max heap)");
    std::cout << "Top element: " << heap_vec.front() << "\n";

    // 9. std::push_heap
    std::cout << "\n9. PUSH_HEAP:\n";
    heap_vec.push_back(10);
    std::push_heap(heap_vec.begin(), heap_vec.end());
    print(heap_vec, "After push_heap(10)");
    std::cout << "New top: " << heap_vec.front() << "\n";

    // 10. std::pop_heap
    std::cout << "\n10. POP_HEAP:\n";
    std::pop_heap(heap_vec.begin(), heap_vec.end());
    int top = heap_vec.back();
    heap_vec.pop_back();
    std::cout << "Popped: " << top << "\n";
    print(heap_vec, "After pop_heap");

    // 11. std::sort_heap
    std::cout << "\n11. SORT_HEAP:\n";
    std::sort_heap(heap_vec.begin(), heap_vec.end());
    print(heap_vec, "After sort_heap (sorted)");

    // 12. std::is_heap
    std::cout << "\n12. IS_HEAP:\n";
    std::vector<int> check_heap = {9, 5, 6, 1, 3, 2, 4};
    bool is_heap = std::is_heap(check_heap.begin(), check_heap.end());
    std::cout << "is_heap: " << (is_heap ? "yes" : "no") << "\n";

    std::make_heap(check_heap.begin(), check_heap.end());
    is_heap = std::is_heap(check_heap.begin(), check_heap.end());
    std::cout << "After make_heap, is_heap: " << (is_heap ? "yes" : "no") << "\n";

    // ========== SET OPERATIONS ==========
    separator("SET OPERATIONS (on sorted ranges)");

    // 13. std::merge
    std::cout << "\n13. MERGE:\n";
    std::vector<int> v1 = {1, 3, 5, 7};
    std::vector<int> v2 = {2, 4, 6, 8};
    std::vector<int> merged;
    std::merge(v1.begin(), v1.end(), v2.begin(), v2.end(),
               std::back_inserter(merged));
    print(merged, "Merged");

    // 14. std::set_union
    std::cout << "\n14. SET_UNION:\n";
    std::vector<int> s1 = {1, 2, 3, 4, 5};
    std::vector<int> s2 = {3, 4, 5, 6, 7};
    std::vector<int> union_result;
    std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(),
                   std::back_inserter(union_result));
    print(union_result, "Union");

    // 15. std::set_intersection
    std::cout << "\n15. SET_INTERSECTION:\n";
    std::vector<int> intersection;
    std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                          std::back_inserter(intersection));
    print(intersection, "Intersection");

    // 16. std::set_difference
    std::cout << "\n16. SET_DIFFERENCE:\n";
    std::vector<int> difference;
    std::set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(),
                        std::back_inserter(difference));
    print(difference, "Difference (s1 - s2)");

    // 17. std::set_symmetric_difference
    std::cout << "\n17. SET_SYMMETRIC_DIFFERENCE:\n";
    std::vector<int> sym_diff;
    std::set_symmetric_difference(s1.begin(), s1.end(), s2.begin(), s2.end(),
                                  std::back_inserter(sym_diff));
    print(sym_diff, "Symmetric difference");

    // 18. std::includes
    std::cout << "\n18. INCLUDES:\n";
    std::vector<int> subset = {2, 3, 4};
    bool is_subset = std::includes(s1.begin(), s1.end(), subset.begin(), subset.end());
    std::cout << "{2,3,4} is subset of s1: " << (is_subset ? "yes" : "no") << "\n";

    // ========== PERMUTATION OPERATIONS ==========
    separator("PERMUTATION OPERATIONS");

    // 19. std::next_permutation
    std::cout << "\n19. NEXT_PERMUTATION:\n";
    std::vector<int> perm = {1, 2, 3};
    std::cout << "All permutations of {1, 2, 3}:\n";
    do {
        print(perm, "");
    } while (std::next_permutation(perm.begin(), perm.end()));

    // 20. std::prev_permutation
    std::cout << "\n20. PREV_PERMUTATION:\n";
    std::vector<int> perm2 = {3, 2, 1};
    std::cout << "Reverse permutations:\n";
    int count = 0;
    do {
        print(perm2, "");
        if (++count >= 3) break;  // Show only first 3
    } while (std::prev_permutation(perm2.begin(), perm2.end()));

    // 21. std::is_permutation
    std::cout << "\n21. IS_PERMUTATION:\n";
    std::vector<int> orig = {1, 2, 3, 4};
    std::vector<int> perm_of_orig = {4, 1, 3, 2};
    std::vector<int> not_perm = {1, 2, 3, 5};

    bool is_perm1 = std::is_permutation(orig.begin(), orig.end(), perm_of_orig.begin());
    bool is_perm2 = std::is_permutation(orig.begin(), orig.end(), not_perm.begin());

    std::cout << "{4,1,3,2} is permutation of {1,2,3,4}: " << (is_perm1 ? "yes" : "no") << "\n";
    std::cout << "{1,2,3,5} is permutation of {1,2,3,4}: " << (is_perm2 ? "yes" : "no") << "\n";

    // ========== OTHER USEFUL ALGORITHMS ==========
    separator("OTHER USEFUL ALGORITHMS");

    // 22. std::shuffle
    std::cout << "\n22. SHUFFLE:\n";
    std::vector<int> shuffle_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffle_vec.begin(), shuffle_vec.end(), g);
    print(shuffle_vec, "Shuffled");

    // 23. std::sample (C++17)
    std::cout << "\n23. SAMPLE:\n";
    std::vector<int> population = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> sample_out;
    std::sample(population.begin(), population.end(),
                std::back_inserter(sample_out), 5, g);
    print(sample_out, "Random sample of 5");

    // 24. std::clamp (C++17)
    std::cout << "\n24. CLAMP:\n";
    int val1 = std::clamp(5, 1, 10);    // 5 (within range)
    int val2 = std::clamp(-5, 1, 10);   // 1 (below min)
    int val3 = std::clamp(15, 1, 10);   // 10 (above max)
    std::cout << "clamp(5, 1, 10) = " << val1 << "\n";
    std::cout << "clamp(-5, 1, 10) = " << val2 << "\n";
    std::cout << "clamp(15, 1, 10) = " << val3 << "\n";

    // 25. std::lexicographical_compare
    std::cout << "\n25. LEXICOGRAPHICAL_COMPARE:\n";
    std::vector<int> lex1 = {1, 2, 3};
    std::vector<int> lex2 = {1, 2, 4};
    std::vector<int> lex3 = {1, 2, 3, 4};

    bool less = std::lexicographical_compare(lex1.begin(), lex1.end(),
                                             lex2.begin(), lex2.end());
    std::cout << "{1,2,3} < {1,2,4}: " << (less ? "yes" : "no") << "\n";

    less = std::lexicographical_compare(lex1.begin(), lex1.end(),
                                        lex3.begin(), lex3.end());
    std::cout << "{1,2,3} < {1,2,3,4}: " << (less ? "yes" : "no") << "\n";

    // 26. std::equal
    std::cout << "\n26. EQUAL:\n";
    std::vector<int> eq1 = {1, 2, 3, 4, 5};
    std::vector<int> eq2 = {1, 2, 3, 4, 5};
    std::vector<int> eq3 = {1, 2, 3, 4, 6};

    bool equal1 = std::equal(eq1.begin(), eq1.end(), eq2.begin());
    bool equal2 = std::equal(eq1.begin(), eq1.end(), eq3.begin());

    std::cout << "eq1 == eq2: " << (equal1 ? "yes" : "no") << "\n";
    std::cout << "eq1 == eq3: " << (equal2 ? "yes" : "no") << "\n";

    // 27. std::mismatch
    std::cout << "\n27. MISMATCH:\n";
    std::vector<int> mm1 = {1, 2, 3, 4, 5};
    std::vector<int> mm2 = {1, 2, 9, 4, 5};

    auto [it1, it2] = std::mismatch(mm1.begin(), mm1.end(), mm2.begin());
    if (it1 != mm1.end()) {
        std::cout << "First mismatch at position " << (it1 - mm1.begin())
                  << ": " << *it1 << " vs " << *it2 << "\n";
    }

    // 28. std::swap_ranges
    std::cout << "\n28. SWAP_RANGES:\n";
    std::vector<int> swap1 = {1, 2, 3};
    std::vector<int> swap2 = {7, 8, 9};

    print(swap1, "Before swap1");
    print(swap2, "Before swap2");

    std::swap_ranges(swap1.begin(), swap1.end(), swap2.begin());

    print(swap1, "After swap1");
    print(swap2, "After swap2");

    // ========== PRACTICAL EXAMPLES ==========
    separator("PRACTICAL EXAMPLES");

    // 29. Find top K elements
    std::cout << "\n29. FIND TOP 3 ELEMENTS:\n";
    std::vector<int> data = {3, 7, 1, 9, 4, 2, 8, 6, 5};
    std::partial_sort(data.begin(), data.begin() + 3, data.end(), std::greater<int>());
    std::cout << "Top 3: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n";

    // 30. Running sum
    std::cout << "\n30. RUNNING SUM:\n";
    std::vector<int> running = {1, 2, 3, 4, 5};
    std::vector<int> running_sum(5);
    std::partial_sum(running.begin(), running.end(), running_sum.begin());
    print(running_sum, "Running sum");

    std::cout << "\n=== END OF ALGORITHMS PART 2 ===\n";

    return 0;
}
