// Example 9: Inline Namespaces
// Used for versioning APIs while maintaining backward compatibility

#include <iostream>

namespace MyLibrary {
    // Old version (version 1)
    namespace v1 {
        void processData() {
            std::cout << "Processing data - Version 1 (old algorithm)" << std::endl;
        }
    }

    // Current version (version 2) - marked as inline
    // Members of inline namespace can be accessed without specifying the namespace
    inline namespace v2 {
        void processData() {
            std::cout << "Processing data - Version 2 (new improved algorithm)" << std::endl;
        }

        void extraFeature() {
            std::cout << "Extra feature only in v2" << std::endl;
        }
    }

    // Future version (version 3) - not inline yet
    namespace v3 {
        void processData() {
            std::cout << "Processing data - Version 3 (experimental)" << std::endl;
        }
    }
}

int main() {
    // Calls v2::processData() because v2 is inline
    MyLibrary::processData();

    // Can explicitly call old version
    MyLibrary::v1::processData();

    // Can explicitly call new experimental version
    MyLibrary::v3::processData();

    // Extra feature from inline namespace
    MyLibrary::extraFeature();

    return 0;
}

// Use case: When you update a library, make the new version inline
// Old code still works, but uses the new version by default
