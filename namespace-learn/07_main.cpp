// Example 7: Namespace Across Multiple Files (Main File)
// This shows how to use namespaces defined in separate files

#include <iostream>
#include "07_geometry.h"

int main() {
    // Use functions from Geometry namespace defined in other files
    double circleArea = Geometry::calculateCircleArea(5.0);
    double rectArea = Geometry::calculateRectangleArea(4.0, 6.0);
    double triArea = Geometry::calculateTriangleArea(3.0, 8.0);

    std::cout << "Circle area: " << circleArea << std::endl;
    std::cout << "Rectangle area: " << rectArea << std::endl;
    std::cout << "Triangle area: " << triArea << std::endl;

    // Use nested namespace
    double sphereVol = Geometry::ThreeD::calculateSphereVolume(3.0);
    double cubeVol = Geometry::ThreeD::calculateCubeVolume(4.0);

    std::cout << "Sphere volume: " << sphereVol << std::endl;
    std::cout << "Cube volume: " << cubeVol << std::endl;

    return 0;
}

// Compile with: g++ 07_main.cpp 07_geometry.cpp -o 07_program
