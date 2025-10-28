/**
 * 06_container_adapters.cpp
 *
 * CONTAINER ADAPTERS - Wrappers around other containers
 * - stack: LIFO (Last In First Out)
 * - queue: FIFO (First In First Out)
 * - priority_queue: Elements sorted by priority
 *
 * Features:
 * - Built on top of other containers (default: deque)
 * - Restricted interface (specific operations only)
 * - No iterators (by design)
 */

#include <iostream>
#include <stack>
#include <queue>
#include <deque>
#include <vector>
#include <list>
#include <functional>
#include <string>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

template<typename T>
void printStack(std::stack<T> s, const std::string& label) {
    std::cout << label << " (top to bottom): ";
    while (!s.empty()) {
        std::cout << s.top() << " ";
        s.pop();
    }
    std::cout << "\n";
}

template<typename T>
void printQueue(std::queue<T> q, const std::string& label) {
    std::cout << label << " (front to back): ";
    while (!q.empty()) {
        std::cout << q.front() << " ";
        q.pop();
    }
    std::cout << "\n";
}

template<typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
void printPriorityQueue(std::priority_queue<T, Container, Compare> pq, const std::string& label) {
    std::cout << label << " (highest to lowest): ";
    while (!pq.empty()) {
        std::cout << pq.top() << " ";
        pq.pop();
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=== CONTAINER ADAPTERS ===\n";

    // ========== STACK ==========
    separator("STACK (LIFO)");

    // 1. Stack Basics
    std::cout << "\n1. STACK INITIALIZATION:\n";
    std::stack<int> stack1;                     // Default (uses deque)
    std::stack<int, std::vector<int>> stack2;   // Using vector
    std::stack<int, std::deque<int>> stack3;    // Explicit deque

    // 2. Push Operations
    std::cout << "\n2. PUSH OPERATIONS:\n";
    std::stack<int> numbers;

    numbers.push(10);
    numbers.push(20);
    numbers.push(30);
    numbers.push(40);

    std::cout << "Pushed: 10, 20, 30, 40\n";
    std::cout << "Size: " << numbers.size() << "\n";
    std::cout << "Top: " << numbers.top() << "\n";

    // Emplace
    numbers.emplace(50);
    std::cout << "After emplace(50), top: " << numbers.top() << "\n";

    // 3. Pop Operations
    std::cout << "\n3. POP OPERATIONS:\n";
    printStack(numbers, "Stack before pops");

    numbers.pop();  // Removes top element
    std::cout << "After pop(), top: " << numbers.top() << "\n";
    std::cout << "Size: " << numbers.size() << "\n";

    // 4. Stack Properties
    std::cout << "\n4. STACK PROPERTIES:\n";
    std::stack<int> check;
    std::cout << "Empty stack - empty(): " << (check.empty() ? "true" : "false") << "\n";

    check.push(1);
    std::cout << "After push(1) - empty(): " << (check.empty() ? "true" : "false") << "\n";
    std::cout << "size(): " << check.size() << "\n";

    // 5. Practical Stack Example: Balanced Parentheses
    std::cout << "\n5. EXAMPLE: BALANCED PARENTHESES CHECKER:\n";

    auto isBalanced = [](const std::string& expr) -> bool {
        std::stack<char> s;
        for (char c : expr) {
            if (c == '(' || c == '{' || c == '[') {
                s.push(c);
            } else if (c == ')' || c == '}' || c == ']') {
                if (s.empty()) return false;
                char top = s.top();
                s.pop();
                if ((c == ')' && top != '(') ||
                    (c == '}' && top != '{') ||
                    (c == ']' && top != '[')) {
                    return false;
                }
            }
        }
        return s.empty();
    };

    std::cout << "\"({[]})\" is " << (isBalanced("({[]})") ? "balanced" : "not balanced") << "\n";
    std::cout << "\"({[}])\" is " << (isBalanced("({[}])") ? "balanced" : "not balanced") << "\n";

    // 6. Stack Use Cases
    std::cout << "\n6. STACK USE CASES:\n";
    std::cout << "- Function call stack\n";
    std::cout << "- Undo/Redo operations\n";
    std::cout << "- Expression evaluation\n";
    std::cout << "- Backtracking algorithms\n";
    std::cout << "- Browser history (back button)\n";

    // ========== QUEUE ==========
    separator("QUEUE (FIFO)");

    // 7. Queue Basics
    std::cout << "\n7. QUEUE INITIALIZATION:\n";
    std::queue<int> queue1;                     // Default (uses deque)
    std::queue<int, std::list<int>> queue2;     // Using list
    std::queue<int, std::deque<int>> queue3;    // Explicit deque

    // 8. Queue Operations
    std::cout << "\n8. QUEUE OPERATIONS:\n";
    std::queue<std::string> tasks;

    tasks.push("Task 1");
    tasks.push("Task 2");
    tasks.push("Task 3");
    tasks.emplace("Task 4");

    std::cout << "Front: " << tasks.front() << "\n";
    std::cout << "Back: " << tasks.back() << "\n";
    std::cout << "Size: " << tasks.size() << "\n";

    printQueue(tasks, "Queue");

    // 9. Processing Queue
    std::cout << "\n9. PROCESSING QUEUE:\n";
    std::queue<int> process_queue;
    for (int i = 1; i <= 5; ++i) {
        process_queue.push(i * 10);
    }

    std::cout << "Processing tasks:\n";
    while (!process_queue.empty()) {
        std::cout << "Processing: " << process_queue.front() << "\n";
        process_queue.pop();
    }

    // 10. Queue Use Cases
    std::cout << "\n10. QUEUE USE CASES:\n";
    std::cout << "- Task scheduling\n";
    std::cout << "- BFS (Breadth-First Search)\n";
    std::cout << "- Print spooler\n";
    std::cout << "- Message queues\n";
    std::cout << "- Customer service lines\n";

    // ========== PRIORITY QUEUE ==========
    separator("PRIORITY QUEUE");

    // 11. Priority Queue Basics
    std::cout << "\n11. PRIORITY QUEUE INITIALIZATION:\n";
    std::priority_queue<int> pq1;               // Max heap (default)
    std::priority_queue<int, std::vector<int>, std::greater<int>> pq2;  // Min heap

    // 12. Max Heap (Default)
    std::cout << "\n12. MAX HEAP (DEFAULT):\n";
    std::priority_queue<int> max_heap;

    max_heap.push(30);
    max_heap.push(10);
    max_heap.push(50);
    max_heap.push(20);
    max_heap.push(40);

    std::cout << "Inserted: 30, 10, 50, 20, 40\n";
    std::cout << "Top (highest): " << max_heap.top() << "\n";

    printPriorityQueue(max_heap, "Max heap");

    // 13. Min Heap
    std::cout << "\n13. MIN HEAP:\n";
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;

    min_heap.push(30);
    min_heap.push(10);
    min_heap.push(50);
    min_heap.push(20);
    min_heap.push(40);

    std::cout << "Inserted: 30, 10, 50, 20, 40\n";
    std::cout << "Top (lowest): " << min_heap.top() << "\n";

    printPriorityQueue(min_heap, "Min heap");

    // 14. Custom Priority
    std::cout << "\n14. CUSTOM PRIORITY:\n";

    struct Task {
        std::string name;
        int priority;

        // For max heap, higher priority value = higher priority
        bool operator<(const Task& other) const {
            return priority < other.priority;
        }
    };

    std::priority_queue<Task> task_queue;

    task_queue.push({"Low Priority Task", 1});
    task_queue.push({"High Priority Task", 10});
    task_queue.push({"Medium Priority Task", 5});
    task_queue.push({"Critical Task", 20});

    std::cout << "Processing tasks by priority:\n";
    while (!task_queue.empty()) {
        Task t = task_queue.top();
        std::cout << "Priority " << t.priority << ": " << t.name << "\n";
        task_queue.pop();
    }

    // 15. Priority Queue with Lambda
    std::cout << "\n15. PRIORITY QUEUE WITH LAMBDA:\n";

    auto compare = [](int a, int b) { return a > b; };  // Min heap
    std::priority_queue<int, std::vector<int>, decltype(compare)> custom_pq(compare);

    custom_pq.push(30);
    custom_pq.push(10);
    custom_pq.push(50);

    std::cout << "Custom PQ (min heap via lambda):\n";
    std::cout << "Top: " << custom_pq.top() << "\n";

    // 16. Priority Queue Operations
    std::cout << "\n16. PRIORITY QUEUE OPERATIONS:\n";
    std::priority_queue<int> ops_pq;

    ops_pq.push(5);
    ops_pq.push(15);
    ops_pq.emplace(25);

    std::cout << "Size: " << ops_pq.size() << "\n";
    std::cout << "Empty: " << (ops_pq.empty() ? "true" : "false") << "\n";
    std::cout << "Top: " << ops_pq.top() << "\n";

    ops_pq.pop();
    std::cout << "After pop(), top: " << ops_pq.top() << "\n";

    // 17. Priority Queue Use Cases
    std::cout << "\n17. PRIORITY QUEUE USE CASES:\n";
    std::cout << "- Dijkstra's algorithm\n";
    std::cout << "- Huffman encoding\n";
    std::cout << "- Job scheduling (OS)\n";
    std::cout << "- Event simulation\n";
    std::cout << "- Finding K largest/smallest elements\n";

    // 18. Practical Example: Top K Elements
    separator("PRACTICAL EXAMPLE");

    std::cout << "\n18. FIND TOP 3 SCORES:\n";
    std::vector<int> scores = {85, 92, 78, 95, 88, 76, 90, 82};

    // Use min heap of size k to find top k elements
    std::priority_queue<int, std::vector<int>, std::greater<int>> top_k_heap;
    int k = 3;

    for (int score : scores) {
        if (top_k_heap.size() < k) {
            top_k_heap.push(score);
        } else if (score > top_k_heap.top()) {
            top_k_heap.pop();
            top_k_heap.push(score);
        }
    }

    std::cout << "Scores: ";
    for (int s : scores) std::cout << s << " ";
    std::cout << "\n";

    std::cout << "Top 3 scores: ";
    while (!top_k_heap.empty()) {
        std::cout << top_k_heap.top() << " ";
        top_k_heap.pop();
    }
    std::cout << "\n";

    // 19. Adapter Comparison
    separator("ADAPTER COMPARISON");

    std::cout << "\nSTACK (LIFO):\n";
    std::cout << "Operations: push(), pop(), top()\n";
    std::cout << "Access: Top element only\n";
    std::cout << "Default: deque\n";

    std::cout << "\nQUEUE (FIFO):\n";
    std::cout << "Operations: push(), pop(), front(), back()\n";
    std::cout << "Access: Front and back only\n";
    std::cout << "Default: deque\n";

    std::cout << "\nPRIORITY_QUEUE:\n";
    std::cout << "Operations: push(), pop(), top()\n";
    std::cout << "Access: Highest priority element only\n";
    std::cout << "Default: vector (max heap)\n";
    std::cout << "Complexity: O(log n) insert/remove\n";

    std::cout << "\n=== END OF CONTAINER ADAPTERS ===\n";

    return 0;
}
