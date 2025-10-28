// Example 4: Namespace Aliases
// Create shorter names for long namespaces

#include <iostream>

namespace VeryLongCompanyNameForSoftwareDevelopment {
    namespace Engineering {
        namespace CloudServices {
            void deployToCloud() {
                std::cout << "Deploying to cloud..." << std::endl;
            }

            void scaleResources() {
                std::cout << "Scaling cloud resources..." << std::endl;
            }
        }
    }
}


int main() {
    // Create an alias for the long namespace
    namespace Cloud = VeryLongCompanyNameForSoftwareDevelopment::Engineering::CloudServices;

    // Now use the shorter alias
    Cloud::deployToCloud();
    Cloud::scaleResources();

    // You can also create multiple aliases
    namespace VLCN = VeryLongCompanyNameForSoftwareDevelopment;
    VLCN::Engineering::CloudServices::deployToCloud();

    return 0;
}
