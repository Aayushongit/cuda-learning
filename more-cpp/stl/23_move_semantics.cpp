/**
 * 23_move_semantics.cpp
 *
 * MOVE SEMANTICS AND PERFECT FORWARDING
 * - Lvalues and rvalues
 * - Move constructors and move assignment
 * - std::move
 * - std::forward
 * - Perfect forwarding
 * - Rule of Five
 */

#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

// Demo class with move semantics
class Buffer {
private:
    int* data;
    size_t size;

public:
    // Constructor
    Buffer(size_t s) : size(s) {
        data = new int[size];
        std::cout << "Buffer(" << size << ") constructed\n";
    }

    // Destructor
    ~Buffer() {
        delete[] data;
        std::cout << "Buffer destroyed\n";
    }

    // Copy constructor
    Buffer(const Buffer& other) : size(other.size) {
        data = new int[size];
        std::copy(other.data, other.data + size, data);
        std::cout << "Buffer copy constructed\n";
    }

    // Copy assignment
    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new int[size];
            std::copy(other.data, other.data + size, data);
            std::cout << "Buffer copy assigned\n";
        }
        return *this;
    }

    // Move constructor
    Buffer(Buffer&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
        std::cout << "Buffer move constructed\n";
    }

    // Move assignment
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
            std::cout << "Buffer move assigned\n";
        }
        return *this;
    }

    size_t get_size() const { return size; }
};

// Process functions for perfect forwarding demonstration
void process(int& x) {
    std::cout << "Lvalue process: " << x << "\n";
}

void process(int&& x) {
    std::cout << "Rvalue process: " << x << "\n";
}

// Perfect forwarding wrapper template
template<typename T>
void wrapper(T&& arg) {
    // Forward arg preserving its value category
    process(std::forward<T>(arg));
}

// Factory pattern with perfect forwarding
template<typename T, typename... Args>
T create(Args&&... args) {
    return T(std::forward<Args>(args)...);
}

int main() {
    std::cout << "=== MOVE SEMANTICS ===\n";

    separator("LVALUE VS RVALUE");

    // 1. Lvalue and Rvalue Basics
    std::cout << "\n1. LVALUE VS RVALUE:\n";
    int x = 10;          // x is lvalue
    int y = 20;          // y is lvalue
    // int z = x + y;    // x + y is rvalue

    std::cout << "Lvalues have names and persistent addresses\n";
    std::cout << "Rvalues are temporaries without names\n";

    // 2. Lvalue References
    std::cout << "\n2. LVALUE REFERENCES:\n";
    int a = 5;
    int& ref = a;        // Lvalue reference
    ref = 10;
    std::cout << "a = " << a << "\n";
    // int& bad_ref = 5; // Error: can't bind lvalue ref to rvalue

    // 3. Rvalue References
    std::cout << "\n3. RVALUE REFERENCES:\n";
    int&& rref = 10;     // Rvalue reference binds to rvalue
    rref = 20;
    std::cout << "rref = " << rref << "\n";
    // int&& bad_rref = a; // Error: can't bind rvalue ref to lvalue

    separator("STD::MOVE");

    // 4. std::move Basics
    std::cout << "\n4. STD::MOVE:\n";
    std::string str1 = "Hello";
    std::string str2 = std::move(str1);  // Move str1 into str2

    std::cout << "str2: " << str2 << "\n";
    std::cout << "str1 (moved-from): '" << str1 << "'\n";
    std::cout << "str1 is valid but unspecified state\n";

    // 5. Move with Vectors
    std::cout << "\n5. MOVE WITH VECTORS:\n";
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::cout << "vec1 size before move: " << vec1.size() << "\n";

    std::vector<int> vec2 = std::move(vec1);
    std::cout << "vec2 size after move: " << vec2.size() << "\n";
    std::cout << "vec1 size after move: " << vec1.size() << "\n";

    separator("MOVE CONSTRUCTORS AND ASSIGNMENT");

    // 6. Custom Class with Move
    std::cout << "\n6. CUSTOM CLASS WITH MOVE:\n";
    {
        Buffer buf1(100);
        Buffer buf2(std::move(buf1));  // Move constructor
        std::cout << "buf2 size: " << buf2.get_size() << "\n";
        std::cout << "buf1 size: " << buf1.get_size() << "\n";
    }

    // 7. Move Assignment
    std::cout << "\n7. MOVE ASSIGNMENT:\n";
    {
        Buffer buf1(100);
        Buffer buf2(50);
        buf2 = std::move(buf1);  // Move assignment
        std::cout << "After move assignment\n";
    }

    separator("PERFECT FORWARDING");

    // 8. std::forward Basics
    std::cout << "\n8. STD::FORWARD:\n";
    auto process = [](auto&& arg) {
        std::cout << "Received argument\n";
    };

    int val = 42;
    process(val);         // Lvalue
    process(42);          // Rvalue

    // 9. Perfect Forwarding Template
    std::cout << "\n9. PERFECT FORWARDING:\n";
    wrapper(val);         // Forwards as lvalue
    wrapper(42);          // Forwards as rvalue

    // 10. Factory Pattern with Perfect Forwarding
    std::cout << "\n10. FACTORY WITH PERFECT FORWARDING:\n";
    auto str = create<std::string>("Hello", 5, '!');
    std::cout << "Created: " << str << "\n";

    separator("RETURN VALUE OPTIMIZATION");

    // 11. RVO (Return Value Optimization)
    std::cout << "\n11. RVO:\n";
    auto make_buffer = []() {
        Buffer buf(200);
        return buf;  // RVO: no copy, no move
    };

    std::cout << "Calling make_buffer:\n";
    Buffer result = make_buffer();

    separator("MOVE IN PRACTICE");

    // 12. Move in Algorithms
    std::cout << "\n12. MOVE IN ALGORITHMS:\n";
    std::vector<std::string> source = {"apple", "banana", "cherry"};
    std::vector<std::string> dest;

    dest.reserve(source.size());
    std::move(source.begin(), source.end(), std::back_inserter(dest));

    std::cout << "Dest: ";
    for (const auto& s : dest) std::cout << s << " ";
    std::cout << "\n";

    std::cout << "Source after move: ";
    for (const auto& s : source) std::cout << "[" << s << "] ";
    std::cout << "\n";

    // 13. Move in Sorting
    std::cout << "\n13. MOVE IN SORTING:\n";
    std::vector<std::string> words = {"zebra", "apple", "mango", "banana"};
    std::sort(words.begin(), words.end());  // Uses move operations

    std::cout << "Sorted: ";
    for (const auto& w : words) std::cout << w << " ";
    std::cout << "\n";

    // 14. emplace vs push
    std::cout << "\n14. EMPLACE VS PUSH:\n";
    std::vector<Buffer> buffers;

    std::cout << "push_back (copy/move):\n";
    Buffer temp(50);
    buffers.push_back(std::move(temp));

    std::cout << "\nemplace_back (construct in-place):\n";
    buffers.emplace_back(100);  // Construct directly

    separator("RULE OF FIVE");

    std::cout << "\n15. RULE OF FIVE:\n";
    std::cout << "If you define any of:\n";
    std::cout << "1. Destructor\n";
    std::cout << "2. Copy constructor\n";
    std::cout << "3. Copy assignment operator\n";
    std::cout << "4. Move constructor\n";
    std::cout << "5. Move assignment operator\n";
    std::cout << "\nThen you should probably define all five.\n";

    separator("NOEXCEPT AND MOVE");

    std::cout << "\n16. NOEXCEPT AND MOVE:\n";
    std::cout << "Mark move operations noexcept:\n";
    std::cout << "- Enables optimizations (e.g., in vector reallocation)\n";
    std::cout << "- Indicates no-throw guarantee\n";
    std::cout << "- Required for strong exception safety\n";

    // 17. Move if noexcept
    std::cout << "\n17. MOVE_IF_NOEXCEPT:\n";
    struct MayThrow {
        MayThrow(MayThrow&&) {}  // Not noexcept
    };

    struct NoThrow {
        NoThrow(NoThrow&&) noexcept {}
    };

    std::cout << "MayThrow is nothrow move constructible: "
              << std::is_nothrow_move_constructible_v<MayThrow> << "\n";
    std::cout << "NoThrow is nothrow move constructible: "
              << std::is_nothrow_move_constructible_v<NoThrow> << "\n";

    separator("BEST PRACTICES");

    std::cout << "\n1. Use std::move when transferring ownership\n";
    std::cout << "2. Use std::forward in forwarding references (T&&)\n";
    std::cout << "3. Mark move operations noexcept\n";
    std::cout << "4. Implement Rule of Five when managing resources\n";
    std::cout << "5. Prefer emplace over push for in-place construction\n";
    std::cout << "6. Don't use moved-from objects (except to destroy/assign)\n";
    std::cout << "7. Return by value (RVO/move will optimize)\n";
    std::cout << "8. Use std::move in return statements only when needed\n";
    std::cout << "9. Move semantics enable efficient transfer of resources\n";
    std::cout << "10. Perfect forwarding preserves value categories\n";

    std::cout << "\n=== END OF MOVE SEMANTICS ===\n";

    return 0;
}
