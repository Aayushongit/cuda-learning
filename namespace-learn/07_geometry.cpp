// Example 7: Namespace Across Multiple Files (Implementation File)
// This implements the functions declared in the header

#include "07_geometry.h"

// Implement functions in the Geometry namespace
namespace Geometry {
    const double PI = 3.14159265359;

    double calculateCircleArea(double radius) {
        return PI * radius * radius;
    }

    double calculateRectangleArea(double width, double height) {
        return width * height;
    }

    double calculateTriangleArea(double base, double height) {
        return 0.5 * base * height;
    }

    // Implement nested namespace functions
    namespace ThreeD {
        double calculateSphereVolume(double radius) {
            return (4.0 / 3.0) * PI * radius * radius * radius;
        }

        double calculateCubeVolume(double side) {
            return side * side * side;
        }
    }
}
