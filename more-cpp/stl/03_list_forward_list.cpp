/**
 * 03_list_forward_list.cpp
 *
 * LIST - Doubly-linked list
 * FORWARD_LIST - Singly-linked list
 *
 * LIST Features:
 * - Bidirectional traversal
 * - Fast insertion/deletion anywhere O(1)
 * - No random access
 * - More memory overhead (prev + next pointers)
 *
 * FORWARD_LIST Features:
 * - Forward-only traversal
 * - Less memory overhead (only next pointer)
 * - Fast insertion/deletion O(1)
 * - Most memory-efficient linked list
 */

#include <iostream>
#include <list>
#include <forward_list>
#include <algorithm>

template<typename T>
void printList(const std::list<T>& lst, const std::string& label) {
    std::cout << label << ": ";
    for (const auto& elem : lst) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

template<typename T>
void printForwardList(const std::forward_list<T>& flst, const std::string& label) {
    std::cout << label << ": ";
    for (const auto& elem : flst) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::cout << "=== LIST AND FORWARD_LIST ===\n";

    // ========== STD::LIST ==========
    separator("STD::LIST (DOUBLY-LINKED LIST)");

    // 1. List Initialization
    std::cout << "\n1. LIST INITIALIZATION:\n";
    std::list<int> list1;                       // Empty list
    std::list<int> list2(5);                    // 5 elements, default 0
    std::list<int> list3(5, 100);               // 5 elements, value 100
    std::list<int> list4 = {1, 2, 3, 4, 5};    // Initializer list
    std::list<int> list5(list4);                // Copy constructor

    printList(list3, "list3 (5 elements, 100)");
    printList(list4, "list4");

    // 2. Adding Elements
    std::cout << "\n2. ADDING ELEMENTS:\n";
    std::list<int> numbers;

    numbers.push_back(10);      // Add at end
    numbers.push_back(20);
    numbers.push_front(5);      // Add at front
    numbers.push_front(1);

    printList(numbers, "After push operations");

    numbers.emplace_back(30);   // Construct at end
    numbers.emplace_front(0);   // Construct at front

    printList(numbers, "After emplace operations");

    // Insert at specific position
    auto it = numbers.begin();
    std::advance(it, 3);  // Move iterator 3 positions
    numbers.insert(it, 99);

    printList(numbers, "After insert at position 3");

    // Insert multiple elements
    numbers.insert(numbers.end(), {40, 50, 60});
    printList(numbers, "After inserting multiple");

    // 3. Accessing Elements
    std::cout << "\n3. ACCESSING ELEMENTS:\n";
    std::list<int> access_list = {10, 20, 30, 40, 50};

    std::cout << "front() = " << access_list.front() << "\n";
    std::cout << "back() = " << access_list.back() << "\n";

    // Note: No operator[] or at() for list!
    // Must use iterators for middle elements
    auto iter = access_list.begin();
    std::advance(iter, 2);  // Move to 3rd element
    std::cout << "Element at position 2: " << *iter << "\n";

    // 4. Removing Elements
    std::cout << "\n4. REMOVING ELEMENTS:\n";
    std::list<int> remove_list = {1, 2, 3, 4, 5, 6, 7, 8};
    printList(remove_list, "Original");

    remove_list.pop_front();  // Remove first
    printList(remove_list, "After pop_front()");

    remove_list.pop_back();   // Remove last
    printList(remove_list, "After pop_back()");

    // Remove specific value
    std::list<int> value_list = {1, 2, 3, 2, 4, 2, 5};
    printList(value_list, "Before remove(2)");
    value_list.remove(2);  // Remove all occurrences of 2
    printList(value_list, "After remove(2)");

    // Remove with condition
    std::list<int> cond_list = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    printList(cond_list, "Before remove_if (even)");
    cond_list.remove_if([](int n) { return n % 2 == 0; });  // Remove even numbers
    printList(cond_list, "After remove_if (even)");

    // Erase at iterator
    std::list<int> erase_list = {10, 20, 30, 40, 50};
    auto erase_it = erase_list.begin();
    std::advance(erase_it, 2);
    erase_list.erase(erase_it);
    printList(erase_list, "After erase at position 2");

    // 5. List-Specific Operations
    std::cout << "\n5. LIST-SPECIFIC OPERATIONS:\n";

    // Sort
    std::list<int> sort_list = {5, 2, 8, 1, 9, 3};
    printList(sort_list, "Before sort");
    sort_list.sort();  // List has its own sort (more efficient than std::sort)
    printList(sort_list, "After sort");

    // Reverse
    sort_list.reverse();
    printList(sort_list, "After reverse");

    // Unique (remove consecutive duplicates)
    std::list<int> unique_list = {1, 1, 2, 2, 2, 3, 3, 4, 5, 5};
    printList(unique_list, "Before unique");
    unique_list.unique();
    printList(unique_list, "After unique");

    // 6. Merging Lists
    std::cout << "\n6. MERGING LISTS:\n";
    std::list<int> list_a = {1, 3, 5, 7};
    std::list<int> list_b = {2, 4, 6, 8};

    printList(list_a, "list_a");
    printList(list_b, "list_b");

    list_a.merge(list_b);  // Both must be sorted; list_b becomes empty
    printList(list_a, "After merge");
    std::cout << "list_b size after merge: " << list_b.size() << "\n";

    // 7. Splicing (move elements between lists)
    std::cout << "\n7. SPLICING:\n";
    std::list<int> splice_1 = {1, 2, 3};
    std::list<int> splice_2 = {10, 20, 30, 40};

    printList(splice_1, "splice_1 before");
    printList(splice_2, "splice_2 before");

    // Splice entire list
    auto splice_it = splice_1.begin();
    std::advance(splice_it, 2);  // Position before 3
    splice_1.splice(splice_it, splice_2);  // Move all of splice_2 into splice_1

    printList(splice_1, "splice_1 after");
    printList(splice_2, "splice_2 after (empty)");

    // 8. Iterators
    std::cout << "\n8. ITERATORS:\n";
    std::list<int> iter_list = {10, 20, 30, 40, 50};

    std::cout << "Forward: ";
    for (auto it = iter_list.begin(); it != iter_list.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";

    std::cout << "Reverse: ";
    for (auto it = iter_list.rbegin(); it != iter_list.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";

    // ========== STD::FORWARD_LIST ==========
    separator("STD::FORWARD_LIST (SINGLY-LINKED LIST)");

    // 9. Forward List Initialization
    std::cout << "\n9. FORWARD_LIST INITIALIZATION:\n";
    std::forward_list<int> flist1;                          // Empty
    std::forward_list<int> flist2(5);                       // 5 elements, default 0
    std::forward_list<int> flist3(5, 100);                  // 5 elements, value 100
    std::forward_list<int> flist4 = {1, 2, 3, 4, 5};       // Initializer list

    printForwardList(flist4, "flist4");

    // 10. Adding Elements
    std::cout << "\n10. ADDING TO FORWARD_LIST:\n";
    std::forward_list<int> flist;

    flist.push_front(30);
    flist.push_front(20);
    flist.push_front(10);

    printForwardList(flist, "After push_front");

    // Note: No push_back() in forward_list!

    // Insert after position
    auto flist_it = flist.begin();
    flist.insert_after(flist_it, 15);

    printForwardList(flist, "After insert_after");

    // Emplace after
    flist.emplace_front(5);
    printForwardList(flist, "After emplace_front");

    // 11. Accessing Elements
    std::cout << "\n11. ACCESSING FORWARD_LIST:\n";
    std::forward_list<int> access_flist = {10, 20, 30, 40, 50};

    std::cout << "front() = " << access_flist.front() << "\n";
    // Note: No back(), no operator[], no at()!

    // 12. Removing Elements
    std::cout << "\n12. REMOVING FROM FORWARD_LIST:\n";
    std::forward_list<int> remove_flist = {1, 2, 3, 4, 5, 6};
    printForwardList(remove_flist, "Original");

    remove_flist.pop_front();
    printForwardList(remove_flist, "After pop_front()");

    // Remove specific value
    std::forward_list<int> value_flist = {1, 2, 3, 2, 4, 2, 5};
    printForwardList(value_flist, "Before remove(2)");
    value_flist.remove(2);
    printForwardList(value_flist, "After remove(2)");

    // Remove with condition
    std::forward_list<int> cond_flist = {1, 2, 3, 4, 5, 6, 7, 8};
    printForwardList(cond_flist, "Before remove_if (odd)");
    cond_flist.remove_if([](int n) { return n % 2 == 1; });
    printForwardList(cond_flist, "After remove_if (odd)");

    // Erase after
    std::forward_list<int> erase_flist = {10, 20, 30, 40, 50};
    auto erase_fit = erase_flist.before_begin();  // Special iterator before first
    erase_flist.erase_after(erase_fit);  // Erase first element
    printForwardList(erase_flist, "After erase_after (first element)");

    // 13. Forward List Operations
    std::cout << "\n13. FORWARD_LIST OPERATIONS:\n";

    // Sort
    std::forward_list<int> sort_flist = {5, 2, 8, 1, 9};
    printForwardList(sort_flist, "Before sort");
    sort_flist.sort();
    printForwardList(sort_flist, "After sort");

    // Reverse
    sort_flist.reverse();
    printForwardList(sort_flist, "After reverse");

    // Unique
    std::forward_list<int> unique_flist = {1, 1, 2, 2, 3, 3, 4};
    printForwardList(unique_flist, "Before unique");
    unique_flist.unique();
    printForwardList(unique_flist, "After unique");

    // 14. Merging Forward Lists
    std::cout << "\n14. MERGING FORWARD_LISTS:\n";
    std::forward_list<int> fa = {1, 3, 5};
    std::forward_list<int> fb = {2, 4, 6};

    printForwardList(fa, "fa");
    printForwardList(fb, "fb");

    fa.merge(fb);
    printForwardList(fa, "After merge");

    // 15. Comparison: List vs Forward_List
    separator("LIST VS FORWARD_LIST COMPARISON");

    std::cout << "\nLIST:\n";
    std::cout << "+ Bidirectional traversal\n";
    std::cout << "+ Can insert/delete at any position easily\n";
    std::cout << "+ Has size() method\n";
    std::cout << "- More memory overhead (2 pointers per node)\n";

    std::cout << "\nFORWARD_LIST:\n";
    std::cout << "+ Less memory overhead (1 pointer per node)\n";
    std::cout << "+ Faster for forward-only operations\n";
    std::cout << "+ More cache-friendly\n";
    std::cout << "- Forward-only traversal\n";
    std::cout << "- No size() method (C++17 can use std::distance)\n";
    std::cout << "- No push_back(), no back()\n";

    // 16. Use Cases
    separator("USE CASES");

    std::cout << "\nUse LIST when:\n";
    std::cout << "- Frequent insertion/deletion in middle\n";
    std::cout << "- Need bidirectional traversal\n";
    std::cout << "- Iterator stability is important\n";
    std::cout << "- Need to know size frequently\n";

    std::cout << "\nUse FORWARD_LIST when:\n";
    std::cout << "- Memory is very constrained\n";
    std::cout << "- Only forward traversal needed\n";
    std::cout << "- Implementing algorithms like hash tables\n";
    std::cout << "- Maximum performance for forward operations\n";

    std::cout << "\nUse VECTOR when:\n";
    std::cout << "- Random access needed\n";
    std::cout << "- Cache locality important\n";
    std::cout << "- Mostly append operations\n";

    std::cout << "\n=== END OF LIST AND FORWARD_LIST ===\n";

    return 0;
}
