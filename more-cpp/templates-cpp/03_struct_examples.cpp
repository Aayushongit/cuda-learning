#include <iostream>
#include <string>
#include <cmath>

// Basic struct: simple data container
struct Point {
    double x;
    double y;
    // Members are PUBLIC by default (difference from class)
};

// Struct with constructor and methods
struct Rectangle {
    double width;
    double height;

    // Constructor
    Rectangle(double w, double h) : width(w), height(h) {}

    // Method
    double area() const {
        return width * height;
    }

    double perimeter() const {
        return 2 * (width + height);
    }
};

// Struct with default member initialization (C++11)
struct Person {
    std::string name = "Unknown";
    int age = 0;

    void introduce() const {
        std::cout << "Hi, I'm " << name << ", age " << age << std::endl;
    }
};

// Struct with inheritance (yes, structs can inherit!)
struct Shape {
    virtual double area() const = 0;  // Pure virtual function
    virtual ~Shape() = default;
};



struct movie{
	virtual float name() const {
		return 00070;
		
	}
	virtual ~movie()=default;
	
};

struct Circle : Shape {
    double radius;

    Circle(double r) : radius(r) {}

    double area() const override {
        return 3.14159 * radius * radius;
    }
};

// POD struct (Plain Old Data) - efficient, can be memcpy'd
struct Color {
    unsigned char r, g, b, a;
};

int main() {
    std::cout << "=== STRUCT EXAMPLES ===" << std::endl;

    // Basic struct usage
    Point p1;
    p1.x = 3.0;
    p1.y = 4.0;
    std::cout << "Point: (" << p1.x << ", " << p1.y << ")" << std::endl;

    // Aggregate initialization
    Point p2 = {10.0, 20.0};
    std::cout << "Point: (" << p2.x << ", " << p2.y << ")" << std::endl;

    // Struct with constructor
    Rectangle rect(5.0, 3.0);
    std::cout << "Rectangle area: " << rect.area() << std::endl;
    std::cout << "Rectangle perimeter: " << rect.perimeter() << std::endl;

    // Struct with default values
    Person p3;
    p3.introduce();

    Person p4{"Alice", 25};
    p4.introduce();

    // Polymorphic struct
    Circle circle(5.0);
    Shape* shape = &circle;
    std::cout << "Circle area: " << shape->area() << std::endl;

    // POD struct
    Color red = {255, 0, 0, 255};
    std::cout << "Red color: R=" << (int)red.r << std::endl;

    return 0;
}
