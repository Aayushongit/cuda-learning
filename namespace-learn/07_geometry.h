// Example 7: Namespace Across Multiple Files (Header File)
// This header file declares functions in a namespace

#ifndef GEOMETRY_H
#define GEOMETRY_H

// Declare namespace and its members in header file
namespace Geometry {
    // Function declarations
    double calculateCircleArea(double radius);
    double calculateRectangleArea(double width, double height);
    double calculateTriangleArea(double base, double height);

    // You can also declare nested namespaces
    namespace ThreeD {
        double calculateSphereVolume(double radius);
        double calculateCubeVolume(double side);
    }
}

#endif
