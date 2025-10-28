#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>

// Combining typedef/using with templates and structs

// Type aliases for common patterns
using String = std::string;
template<typename T>
using Ptr = std::shared_ptr<T>;
template<typename T>
using Vec = std::vector<T>;
template<typename K, typename V>
using Map = std::map<K, V>;

// Struct with templates
template<typename T>
struct Point3D {
    T x, y, z;

    Point3D(T x = 0, T y = 0, T z = 0) : x(x), y(y), z(z) {}

    void print() const {
        std::cout << "(" << x << ", " << y << ", " << z << ")" << std::endl;
    }

    T distanceSquared() const {
        return x*x + y*y + z*z;
    }
};

// Typedef within templates
template<typename T>
class Graph {
public:
    // Type aliases inside class template
    using NodeID = int;
    using Weight = T;
    using Edge = std::pair<NodeID, Weight>;
    using AdjacencyList = Map<NodeID, Vec<Edge>>;

private:
    AdjacencyList adjacency;

public:
    void addEdge(NodeID from, NodeID to, Weight weight) {
        adjacency[from].push_back({to, weight});
    }

    void printGraph() const {
        for (const auto& [node, edges] : adjacency) {
            std::cout << "Node " << node << ": ";
            for (const auto& [dest, weight] : edges) {
                std::cout << "(" << dest << ", " << weight << ") ";
            }
            std::cout << std::endl;
        }
    }
};

// Template struct with type traits
template<typename T>
struct TypeInfo {
    using ValueType = T;
    using Reference = T&;
    using ConstReference = const T&;
    using Pointer = T*;

    static void printInfo() {
        std::cout << "Size: " << sizeof(T) << " bytes" << std::endl;
    }
};

// Specialized struct template
template<typename T>
struct Storage {
    T data;

    Storage(T value) : data(value) {
        std::cout << "Generic storage created" << std::endl;
    }
};

template<>
struct Storage<String> {
    String data;

    Storage(String value) : data(value) {
        std::cout << "Specialized string storage created: " << data << std::endl;
    }
};

// Complex nested type with using
template<typename Key, typename Value>
class Cache {
public:
    using KeyType = Key;
    using ValueType = Value;
    using PtrType = Ptr<Value>;
    using StorageType = Map<Key, PtrType>;

private:
    StorageType storage;

public:
    void put(Key key, Ptr<Value> value) {
        storage[key] = value;
    }

    PtrType get(Key key) {
        auto it = storage.find(key);
        return (it != storage.end()) ? it->second : nullptr;
    }

    size_t size() const { return storage.size(); }
};

// Function template using all concepts
template<typename Container, typename Func>
void forEach(const Container& container, Func func) {
    for (const auto& item : container) {
        func(item);
    }
}

int main() {
    std::cout << "=== COMBINING ALL CONCEPTS ===" << std::endl;

    // Using type aliases with templates
    Vec<int> numbers = {1, 2, 3, 4, 5};
    Map<String, int> ages = {{"Alice", 25}, {"Bob", 30}};

    std::cout << "Vector size: " << numbers.size() << std::endl;
    std::cout << "Map size: " << ages.size() << std::endl;

    std::cout << std::endl;

    // Template struct
    Point3D<int> p1(1, 2, 3);
    Point3D<double> p2(1.5, 2.5, 3.5);

    std::cout << "Point3D<int>: ";
    p1.print();
    std::cout << "Point3D<double>: ";
    p2.print();

    std::cout << std::endl;

    // Graph with nested type aliases
    Graph<double> graph;
    graph.addEdge(0, 1, 5.5);
    graph.addEdge(0, 2, 3.2);
    graph.addEdge(1, 2, 1.8);

    std::cout << "Graph structure:" << std::endl;
    graph.printGraph();

    std::cout << std::endl;

    // Type info
    std::cout << "Type info for int: ";
    TypeInfo<int>::printInfo();
    std::cout << "Type info for double: ";
    TypeInfo<double>::printInfo();

    std::cout << std::endl;

    // Specialized storage
    Storage<int> intStore(42);
    Storage<String> strStore("Hello Specialization");

    std::cout << std::endl;

    // Cache with smart pointers
    Cache<String, int> cache;
    cache.put("answer", std::make_shared<int>(42));
    cache.put("pi_x100", std::make_shared<int>(314));

    auto answer = cache.get("answer");
    if (answer) {
        std::cout << "Cache hit for 'answer': " << *answer << std::endl;
    }

    std::cout << "Cache size: " << cache.size() << std::endl;

    std::cout << std::endl;

    // Using forEach template function
    Vec<String> names = {"Alice", "Bob", "Charlie"};
    std::cout << "Names: ";
    forEach(names, [](const String& name) {
        std::cout << name << " ";
    });
    std::cout << std::endl;

    return 0;
}


/*
make -j4

make -j$(nproc)


*/