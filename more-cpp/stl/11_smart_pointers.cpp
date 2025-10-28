/**
 * 11_smart_pointers.cpp
 *
 * SMART POINTERS - Automatic memory management
 * - unique_ptr: Exclusive ownership
 * - shared_ptr: Shared ownership (reference counted)
 * - weak_ptr: Non-owning reference to shared_ptr
 * - make_unique, make_shared
 *
 * Benefits:
 * - Automatic memory deallocation
 * - Exception safe
 * - Prevent memory leaks
 * - Clear ownership semantics
 */

#include <iostream>
#include <memory>
#include <vector>
#include <string>

void separator(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

class Resource {
public:
    std::string name;
    Resource(std::string n) : name(n) {
        std::cout << "Resource '" << name << "' created\n";
    }
    ~Resource() {
        std::cout << "Resource '" << name << "' destroyed\n";
    }
    void use() {
        std::cout << "Using resource '" << name << "'\n";
    }
};

int main() {
    std::cout << "=== SMART POINTERS ===\n";

    // ========== UNIQUE_PTR ==========
    separator("UNIQUE_PTR (Exclusive Ownership)");

    // 1. Creating unique_ptr
    std::cout << "\n1. CREATING UNIQUE_PTR:\n";
    {
        std::unique_ptr<int> ptr1(new int(42));
        std::cout << "ptr1 value: " << *ptr1 << "\n";

        // Preferred: make_unique (C++14)
        auto ptr2 = std::make_unique<int>(100);
        std::cout << "ptr2 value: " << *ptr2 << "\n";

        auto res = std::make_unique<Resource>("R1");
        res->use();
    }  // Automatic cleanup
    std::cout << "Scope ended, resources cleaned up\n";

    // 2. unique_ptr Operations
    std::cout << "\n2. UNIQUE_PTR OPERATIONS:\n";
    auto ptr = std::make_unique<int>(50);

    std::cout << "Value: " << *ptr << "\n";
    std::cout << "Address: " << ptr.get() << "\n";

    // Check if valid
    if (ptr) {
        std::cout << "ptr is valid\n";
    }

    // Release ownership (returns raw pointer)
    int* raw = ptr.release();
    std::cout << "After release, ptr is " << (ptr ? "valid" : "null") << "\n";
    std::cout << "Raw pointer value: " << *raw << "\n";
    delete raw;  // Must manually delete

    // Reset (delete old, optionally assign new)
    ptr.reset(new int(75));
    std::cout << "After reset, value: " << *ptr << "\n";

    ptr.reset();  // Delete and set to null
    std::cout << "After reset(), ptr is " << (ptr ? "valid" : "null") << "\n";

    // 3. unique_ptr Cannot be Copied
    std::cout << "\n3. UNIQUE_PTR OWNERSHIP:\n";
    auto original = std::make_unique<Resource>("Original");

    // auto copy = original;  // ERROR: Cannot copy

    // But can be moved
    auto moved = std::move(original);
    std::cout << "After move:\n";
    std::cout << "  original is " << (original ? "valid" : "null") << "\n";
    std::cout << "  moved is " << (moved ? "valid" : "null") << "\n";

    // 4. unique_ptr with Arrays
    std::cout << "\n4. UNIQUE_PTR WITH ARRAYS:\n";
    {
        auto arr = std::make_unique<int[]>(5);
        for (int i = 0; i < 5; ++i) {
            arr[i] = i * 10;
        }

        std::cout << "Array: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << "\n";
    }  // Automatically deletes array

    // 5. unique_ptr with Custom Deleter
    std::cout << "\n5. UNIQUE_PTR WITH CUSTOM DELETER:\n";
    {
        auto custom_deleter = [](int* p) {
            std::cout << "Custom deleter called for value: " << *p << "\n";
            delete p;
        };

        std::unique_ptr<int, decltype(custom_deleter)> ptr(new int(99), custom_deleter);
    }  // Custom deleter invoked

    // 6. unique_ptr in Containers
    std::cout << "\n6. UNIQUE_PTR IN CONTAINERS:\n";
    std::vector<std::unique_ptr<Resource>> resources;
    resources.push_back(std::make_unique<Resource>("V1"));
    resources.push_back(std::make_unique<Resource>("V2"));
    resources.push_back(std::make_unique<Resource>("V3"));

    std::cout << "Using resources in vector:\n";
    for (const auto& r : resources) {
        r->use();
    }
    // Vector cleanup destroys all resources

    // ========== SHARED_PTR ==========
    separator("SHARED_PTR (Shared Ownership)");

    // 7. Creating shared_ptr
    std::cout << "\n7. CREATING SHARED_PTR:\n";
    {
        std::shared_ptr<int> sp1(new int(42));
        std::cout << "sp1 value: " << *sp1 << "\n";

        // Preferred: make_shared (more efficient)
        auto sp2 = std::make_shared<int>(100);
        std::cout << "sp2 value: " << *sp2 << "\n";

        auto res = std::make_shared<Resource>("Shared");
        res->use();
    }
    std::cout << "Scope ended\n";

    // 8. Reference Counting
    std::cout << "\n8. REFERENCE COUNTING:\n";
    auto sp1 = std::make_shared<Resource>("Counted");
    std::cout << "sp1 use_count: " << sp1.use_count() << "\n";

    {
        auto sp2 = sp1;  // Copy increases count
        std::cout << "After copy, use_count: " << sp1.use_count() << "\n";

        auto sp3 = sp1;
        std::cout << "After another copy, use_count: " << sp1.use_count() << "\n";
    }  // sp2 and sp3 go out of scope

    std::cout << "After scope, use_count: " << sp1.use_count() << "\n";
    // Resource destroyed when last shared_ptr is destroyed

    // 9. Copying and Moving shared_ptr
    std::cout << "\n9. COPYING AND MOVING:\n";
    auto original_sp = std::make_shared<int>(50);
    std::cout << "Original use_count: " << original_sp.use_count() << "\n";

    // Copy
    auto copy_sp = original_sp;
    std::cout << "After copy, use_count: " << original_sp.use_count() << "\n";

    // Move
    auto moved_sp = std::move(copy_sp);
    std::cout << "After move:\n";
    std::cout << "  original use_count: " << original_sp.use_count() << "\n";
    std::cout << "  copy_sp is " << (copy_sp ? "valid" : "null") << "\n";

    // 10. shared_ptr Operations
    std::cout << "\n10. SHARED_PTR OPERATIONS:\n";
    auto sp = std::make_shared<int>(77);

    std::cout << "Value: " << *sp << "\n";
    std::cout << "use_count: " << sp.use_count() << "\n";
    std::cout << "unique: " << (sp.unique() ? "yes" : "no") << "\n";

    // Reset
    sp.reset(new int(88));
    std::cout << "After reset, value: " << *sp << "\n";

    // Get raw pointer (use with caution!)
    int* raw_ptr = sp.get();
    std::cout << "Raw pointer value: " << *raw_ptr << "\n";

    // 11. shared_ptr with Arrays (C++17)
    std::cout << "\n11. SHARED_PTR WITH ARRAYS:\n";
    {
        std::shared_ptr<int[]> arr(new int[5]);
        for (int i = 0; i < 5; ++i) {
            arr[i] = i * 20;
        }

        std::cout << "Array: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << "\n";
    }

    // 12. Circular Reference Problem
    std::cout << "\n12. CIRCULAR REFERENCE PROBLEM:\n";
    struct Node {
        int value;
        std::shared_ptr<Node> next;
        std::weak_ptr<Node> prev;  // Use weak_ptr to break cycle

        Node(int v) : value(v) {
            std::cout << "Node " << value << " created\n";
        }
        ~Node() {
            std::cout << "Node " << value << " destroyed\n";
        }
    };

    {
        auto node1 = std::make_shared<Node>(1);
        auto node2 = std::make_shared<Node>(2);

        node1->next = node2;
        node2->prev = node1;  // weak_ptr doesn't increase count

        std::cout << "node1 use_count: " << node1.use_count() << "\n";
        std::cout << "node2 use_count: " << node2.use_count() << "\n";
    }
    std::cout << "Nodes properly destroyed\n";

    // ========== WEAK_PTR ==========
    separator("WEAK_PTR (Non-owning Reference)");

    // 13. weak_ptr Basics
    std::cout << "\n13. WEAK_PTR BASICS:\n";
    std::weak_ptr<Resource> wp;

    {
        auto sp = std::make_shared<Resource>("Temporary");
        wp = sp;  // weak_ptr doesn't increase ref count

        std::cout << "Inside scope:\n";
        std::cout << "  sp use_count: " << sp.use_count() << "\n";
        std::cout << "  wp expired: " << (wp.expired() ? "yes" : "no") << "\n";

        // Lock to get shared_ptr
        if (auto locked = wp.lock()) {
            std::cout << "  locked successfully\n";
            locked->use();
        }
    }

    std::cout << "Outside scope:\n";
    std::cout << "  wp expired: " << (wp.expired() ? "yes" : "no") << "\n";

    // 14. weak_ptr Use Case: Cache
    std::cout << "\n14. WEAK_PTR CACHE PATTERN:\n";

    class Cache {
        std::weak_ptr<Resource> cached;
    public:
        std::shared_ptr<Resource> get(const std::string& name) {
            if (auto sp = cached.lock()) {
                std::cout << "Cache hit!\n";
                return sp;
            }

            std::cout << "Cache miss, creating new resource\n";
            auto sp = std::make_shared<Resource>(name);
            cached = sp;
            return sp;
        }
    };

    Cache cache;
    {
        auto r1 = cache.get("CachedResource");
        auto r2 = cache.get("CachedResource");  // Cache hit
        std::cout << "Both pointers same: " << (r1 == r2 ? "yes" : "no") << "\n";
    }

    auto r3 = cache.get("CachedResource");  // Cache miss (resource was destroyed)

    // ========== BEST PRACTICES ==========
    separator("BEST PRACTICES");

    std::cout << "\n1. Prefer make_unique/make_shared\n";
    std::cout << "   - Exception safe\n";
    std::cout << "   - More efficient (single allocation for shared_ptr)\n";

    std::cout << "\n2. Use unique_ptr by default\n";
    std::cout << "   - Lighter weight\n";
    std::cout << "   - Clear ownership\n";
    std::cout << "   - Can convert to shared_ptr if needed\n";

    std::cout << "\n3. Use shared_ptr for shared ownership\n";
    std::cout << "   - Multiple owners needed\n";
    std::cout << "   - Lifetime managed by reference counting\n";

    std::cout << "\n4. Use weak_ptr to break cycles\n";
    std::cout << "   - Observer pattern\n";
    std::cout << "   - Cache implementations\n";
    std::cout << "   - Parent-child relationships\n";

    std::cout << "\n5. Avoid raw pointers for ownership\n";
    std::cout << "   - Use raw pointers only for non-owning references\n";

    // ========== PRACTICAL EXAMPLE ==========
    separator("PRACTICAL EXAMPLE");

    std::cout << "\n15. FACTORY PATTERN:\n";

    class Shape {
    public:
        virtual ~Shape() = default;
        virtual void draw() = 0;
    };

    class Circle : public Shape {
    public:
        void draw() override { std::cout << "Drawing Circle\n"; }
    };

    class Square : public Shape {
    public:
        void draw() override { std::cout << "Drawing Square\n"; }
    };

    class ShapeFactory {
    public:
        static std::unique_ptr<Shape> createShape(const std::string& type) {
            if (type == "circle") {
                return std::make_unique<Circle>();
            } else if (type == "square") {
                return std::make_unique<Square>();
            }
            return nullptr;
        }
    };

    auto shape1 = ShapeFactory::createShape("circle");
    auto shape2 = ShapeFactory::createShape("square");

    if (shape1) shape1->draw();
    if (shape2) shape2->draw();

    std::cout << "\n=== END OF SMART POINTERS ===\n";

    return 0;
}
