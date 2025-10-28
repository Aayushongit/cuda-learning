// Example 2: Nested Namespaces
// Namespaces can be nested inside other namespaces

#include <iostream>

// Traditional way of nesting namespaces
namespace Company {
    namespace Engineering {
        namespace Backend {
            void deployServer() {
                std::cout << "Deploying backend server..." << std::endl;
            }
        }

        namespace Frontend {
            void deployWebsite() {
                std::cout << "Deploying frontend website..." << std::endl;
            }
        }
    }
}

// C++17 simplified nested namespace syntax
namespace Project::Module::Submodule {
    void execute() {
        std::cout << "Executing submodule..." << std::endl;
    }
}

int main() {
    // Access nested namespace members
    Company::Engineering::Backend::deployServer();
    Company::Engineering::Frontend::deployWebsite();

    Project::Module::Submodule::execute();

    return 0;
}
