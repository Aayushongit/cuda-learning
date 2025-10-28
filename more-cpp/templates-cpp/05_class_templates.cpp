#include <iostream>
#include <string>
#include <stdexcept>

// Basic class template: generic container
template<typename T>

class Box {

private:

    T value;

public:

    Box(T val) : value(val) {}

    void setValue(T val) { value = val; }
    T getValue() const { return value; }

    void print() const {
        std::cout << "Box contains: " << value << std::endl;
    }
    
};

// Class template with multiple parameters
template<typename K, typename V>
class Pair {
private:
    K key;
    V value;

public:
    Pair(K k, V v) : key(k), value(v) {}

    K getKey() const { return key; }
    V getValue() const { return value; }

    void print() const {
        std::cout << key << " => " << value << std::endl;
    }
};

// Class template with non-type parameter
template<typename T, int CAPACITY>
class StaticArray {
private:
    T data[CAPACITY];
    int size;

public:
    StaticArray() : size(0) {}

    void push(T value) {
        if (size >= CAPACITY) {
            throw std::overflow_error("Array is full");
        }
        data[size++] = value;
    }

    T& operator[](int index) {
        if (index < 0 || index >= size) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    int getSize() const { return size; }
    int getCapacity() const { return CAPACITY; }
};

// Template with default parameter
template<typename T, typename Allocator = std::allocator<T>>
class SimpleVector {
private:
    T* data;
    size_t size;

public:
    SimpleVector(size_t n, T val) : size(n) {
        data = new T[size];
        for (size_t i = 0; i < size; i++) {
            data[i] = val;
        }
    }

    ~SimpleVector() {
        delete[] data;
    }

    T& operator[](size_t index) { return data[index]; }
    size_t getSize() const { return size; }
};

// Nested templates
template<typename T>
class Stack {
private:
    template<typename U>
    struct Node {
        U data;
        Node<U>* next;
        Node(U val) : data(val), next(nullptr) {}
    };

    Node<T>* top;
    int size;

public:
    Stack() : top(nullptr), size(0) {}

    void push(T value) {
        Node<T>* newNode = new Node<T>(value);
        newNode->next = top;
        top = newNode;
        size++;
    }

    T pop() {
        if (!top) throw std::underflow_error("Stack is empty");
        Node<T>* temp = top;
        T value = top->data;
        top = top->next;
        delete temp;
        size--;
        return value;
    }

    bool isEmpty() const { return top == nullptr; }
    int getSize() const { return size; }

    ~Stack() {
        while (!isEmpty()) pop();
    }
};

int main() {
    std::cout << "=== CLASS TEMPLATE EXAMPLES ===" << std::endl;

    // Basic template class
    Box<int> intBox(42);
    intBox.print();

    Box<std::string> strBox("Hello Templates");
    strBox.print();

    Box<double> doubleBox(3.14159);
    doubleBox.print();

    // Multiple template parameters
    Pair<std::string, int> age("Alice", 25);
    age.print();

    Pair<int, double> measurement(100, 98.6);
    measurement.print();

    // Non-type template parameter
    StaticArray<int, 5> arr;
    arr.push(10);
    arr.push(20);
    arr.push(30);
    std::cout << "StaticArray[1]: " << arr[1] << std::endl;
    std::cout << "Size: " << arr.getSize() << "/" << arr.getCapacity() << std::endl;

    // Template with default parameter
    SimpleVector<int> vec(3, 99);
    std::cout << "SimpleVector[0]: " << vec[0] << std::endl;

    // Nested template (Stack)
    Stack<std::string> stringStack;
    stringStack.push("First");
    stringStack.push("Second");
    stringStack.push("Third");

    std::cout << "Stack popping: ";
    while (!stringStack.isEmpty()) {
        std::cout << stringStack.pop() << " ";
    }
    std::cout << std::endl;

    return 0;
}
