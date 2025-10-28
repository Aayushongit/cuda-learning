#include <iostream>
#include <vector>
#include <map>
#include <functional>

// using: Modern C++11 way (cleaner, more readable)
using ulong = unsigned long;
using IntPtr = int*;
using IntVector = std::vector<int>;

// using with function types (much cleaner than typedef!)
using FunctionPtr = void(*)(int, int);
using Callback = std::function<void(int)>;

// using with complex types (more readable left-to-right)
using StringToIntVectorMap = std::map<std::string, std::vector<int>>;

// using is especially powerful with templates (typedef can't do this!)
template<typename T>
using Vec = std::vector<T>;

template<typename K, typename V>   
using Map = std::map<K, V>;

void printValue(int x) {
    std::cout << "Value: " << x << std::endl;
}

int main() {
    std::cout << "=== USING EXAMPLES ===" << std::endl;

    // Basic type alias
    ulong bigNum = 123456789UL;
    std::cout << "ulong: " << bigNum << std::endl;

    // Template alias (typedef cannot do this!)
    Vec<int> intVec = {1, 2, 3, 4, 5};
    Vec<std::string> strVec = {"hello", "world"};
    Vec<double> doubleVec = {1.1, 2.2, 3.3};

    std::cout << "Vec<int> size: " << intVec.size() << std::endl;
    std::cout << "Vec<string> size: " << strVec.size() << std::endl;

    // Template alias with multiple parameters
    Map<std::string, int> ages;
    ages["Alice"] = 25;
    ages["Bob"] = 30;
    std::cout << "Alice's age: " << ages["Alice"] << std::endl;

    // Function type alias with std::function
    Callback cb = printValue;
    cb(42);

    return 0;
}
