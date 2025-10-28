/*
 * C++ ENUMS TUTORIAL
 * ==================
 * This file demonstrates different types of enums in C++ and their usage.
 *
 * Topics covered:
 * 1. Traditional C-style enums
 * 2. Enum class (scoped enums) - C++11
 * 3. Specifying underlying type
 * 4. Using enums in switch statements
 * 5. Converting between enums and integers
 * 6. Enum class with explicit values
 */

#include <iostream>
#include <string>
#include <cstdint>

using namespace std;

// ============================================================================
// 1. TRADITIONAL C-STYLE ENUM
// ============================================================================
// Traditional enums are not type-safe and their values are in the global scope
// This can lead to naming conflicts!

enum Color {
    RED,      // 0 by default
    GREEN,    // 1
    BLUE,     // 2
    YELLOW    // 3
};

// You can also specify explicit values
enum Season {
    SPRING = 1,
    SUMMER = 2,
    FALL = 3,
    WINTER = 4
};

// Problem with traditional enums: Values leak into surrounding scope
enum Status {
    SUCCESS = 0,
    FAILURE = 1,
    PENDING = 2
};

// ============================================================================
// 2. ENUM CLASS (SCOPED ENUMS) - C++11
// ============================================================================
// Enum classes are type-safe and their values don't leak into surrounding scope
// This is the RECOMMENDED way to use enums in modern C++

enum class Direction {
    NORTH,
    SOUTH,
    EAST,
    WEST
};

enum class TrafficLight {
    RED,      // No conflict with Color::RED!
    YELLOW,   // No conflict with Color::YELLOW!
    GREEN
};

// ============================================================================
// 3. SPECIFYING UNDERLYING TYPE
// ============================================================================
// You can specify what integer type the enum uses (default is int)

enum class Priority : std::uint8_t {
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
    CRITICAL = 4
};

enum class ErrorCode : int {
    NONE = 0,
    FILE_NOT_FOUND = -1,
    PERMISSION_DENIED = -2,
    INVALID_INPUT = -3
};

// ============================================================================
// 4. ENUM CLASS WITH EXPLICIT BIT FLAGS
// ============================================================================
// Useful for options that can be combined

enum class FilePermission : unsigned int {
    NONE = 0,
    READ = 1,        // 0001
    WRITE = 2,       // 0010
    EXECUTE = 4,     // 0100
    ALL = 7          // 0111 (READ | WRITE | EXECUTE)
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Convert traditional enum to string
string colorToString(Color c) {
    switch(c) {
        case RED: return "Red";
        case GREEN: return "Green";
        case BLUE: return "Blue";
        case YELLOW: return "Yellow";
        default: return "Unknown";
    }
}

// Convert enum class to string
string directionToString(Direction d) {
    switch(d) {
        case Direction::NORTH: return "North";
        case Direction::SOUTH: return "South";
        case Direction::EAST: return "East";
        case Direction::WEST: return "West";
        default: return "Unknown";
    }
}

// Get opposite direction
Direction getOpposite(Direction d) {
    switch(d) {
        case Direction::NORTH: return Direction::SOUTH;
        case Direction::SOUTH: return Direction::NORTH;
        case Direction::EAST: return Direction::WEST;
        case Direction::WEST: return Direction::EAST;
        default: return d;
    }
}

// Traffic light logic
TrafficLight nextLight(TrafficLight current) {
    switch(current) {
        case TrafficLight::RED: return TrafficLight::GREEN;
        case TrafficLight::GREEN: return TrafficLight::YELLOW;
        case TrafficLight::YELLOW: return TrafficLight::RED;
        default: return current;
    }
}

// Priority comparison
string priorityMessage(Priority p) {
    switch(p) {
        case Priority::LOW:
            return "Handle when convenient";
        case Priority::MEDIUM:
            return "Address soon";
        case Priority::HIGH:
            return "Urgent - handle immediately";
        case Priority::CRITICAL:
            return "CRITICAL - drop everything!";
        default:
            return "Unknown priority";
    }
}

// ============================================================================
// DEMONSTRATION FUNCTIONS
// ============================================================================

void demonstrateTraditionalEnums() {
    cout << "\n=== TRADITIONAL ENUMS ===\n";

    // Can use values directly without scope
    Color favoriteColor = RED;
    cout << "Favorite color: " << colorToString(favoriteColor) << endl;

    // Can implicitly convert to int
    cout << "Color value: " << favoriteColor << endl;

    // Can compare with integers (not type-safe!)
    if (favoriteColor == 0) {
        cout << "Color is RED (0)" << endl;
    }

    // Problem: Values leak into scope
    Season currentSeason = SUMMER;
    cout << "Current season: " << currentSeason << endl;

    // This shows the problem with traditional enums
    cout << "\nProblems with traditional enums:" << endl;
    cout << "- Can accidentally compare different enum types" << endl;
    cout << "- Values pollute the surrounding namespace" << endl;
}

void demonstrateEnumClass() {
    cout << "\n=== ENUM CLASS (RECOMMENDED) ===\n";

    // Must use scope to access values
    Direction heading = Direction::NORTH;
    cout << "Heading: " << directionToString(heading) << endl;

    Direction opposite = getOpposite(heading);
    cout << "Opposite direction: " << directionToString(opposite) << endl;

    // Cannot implicitly convert to int (type-safe!)
    // cout << heading << endl;  // ERROR: won't compile!

    // Must explicitly cast if needed
    cout << "Direction value: " << static_cast<int>(heading) << endl;

    // No naming conflicts!
    TrafficLight light = TrafficLight::RED;
    Color carColor = RED;  // Both RED can coexist!

    cout << "\nBenefits of enum class:" << endl;
    cout << "- Type-safe (no accidental comparisons)" << endl;
    cout << "- No namespace pollution" << endl;
    cout << "- Can have same names in different enums" << endl;
}

void demonstrateTrafficLightSimulation() {
    cout << "\n=== TRAFFIC LIGHT SIMULATION ===\n";

    TrafficLight light = TrafficLight::RED;

    for (int i = 0; i < 6; i++) {
        switch(light) {
            case TrafficLight::RED:
                cout << "STOP - Light is RED" << endl;
                break;
            case TrafficLight::YELLOW:
                cout << "CAUTION - Light is YELLOW" << endl;
                break;
            case TrafficLight::GREEN:
                cout << "GO - Light is GREEN" << endl;
                break;
        }
        light = nextLight(light);
    }
}

void demonstratePrioritySystems() {
    cout << "\n=== PRIORITY SYSTEM ===\n";

    Priority tasks[] = {
        Priority::LOW,
        Priority::MEDIUM,
        Priority::HIGH,
        Priority::CRITICAL
    };

    for (Priority p : tasks) {
        // Explicit cast to underlying type
        int priorityValue = static_cast<int>(p);
        cout << "Priority " << priorityValue << ": "
             << priorityMessage(p) << endl;
    }
}

void demonstrateUnderlyingTypes() {
    cout << "\n=== UNDERLYING TYPES ===\n";

    Priority p = Priority::HIGH;
    cout << "Priority uses uint8_t: " << sizeof(p) << " byte(s)" << endl;

    ErrorCode err = ErrorCode::FILE_NOT_FOUND;
    cout << "ErrorCode uses int: " << sizeof(err) << " byte(s)" << endl;
    cout << "Error value: " << static_cast<int>(err) << endl;
}

void demonstrateBitFlags() {
    cout << "\n=== BIT FLAGS (ADVANCED) ===\n";

    // Individual permissions
    FilePermission readOnly = FilePermission::READ;
    FilePermission readWrite = static_cast<FilePermission>(
        static_cast<unsigned int>(FilePermission::READ) |
        static_cast<unsigned int>(FilePermission::WRITE)
    );

    cout << "Read only: " << static_cast<unsigned int>(readOnly) << endl;
    cout << "Read+Write: " << static_cast<unsigned int>(readWrite) << endl;
    cout << "All permissions: " << static_cast<unsigned int>(FilePermission::ALL) << endl;

    // Check if a permission is set
    unsigned int perms = static_cast<unsigned int>(readWrite);
    bool hasRead = (perms & static_cast<unsigned int>(FilePermission::READ)) != 0;
    bool hasExecute = (perms & static_cast<unsigned int>(FilePermission::EXECUTE)) != 0;

    cout << "Has READ permission: " << (hasRead ? "Yes" : "No") << endl;
    cout << "Has EXECUTE permission: " << (hasExecute ? "Yes" : "No") << endl;
}

void demonstrateComparison() {
    cout << "\n=== ENUM COMPARISONS ===\n";

    Direction d1 = Direction::NORTH;
    Direction d2 = Direction::NORTH;
    Direction d3 = Direction::SOUTH;

    cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << endl;
    cout << "d1 == d3: " << (d1 == d3 ? "true" : "false") << endl;
    cout << "d1 != d3: " << (d1 != d3 ? "true" : "false") << endl;

    // Enums can be used in conditional statements
    if (d1 == Direction::NORTH) {
        cout << "Heading north!" << endl;
    }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    cout << "================================================\n";
    cout << "    C++ ENUMS COMPREHENSIVE TUTORIAL\n";
    cout << "================================================\n";

    demonstrateTraditionalEnums();
    demonstrateEnumClass();
    demonstrateTrafficLightSimulation();
    demonstratePrioritySystems();
    demonstrateUnderlyingTypes();
    demonstrateBitFlags();
    demonstrateComparison();

    cout << "\n================================================\n";
    cout << "KEY TAKEAWAYS:\n";
    cout << "================================================\n";
    cout << "1. Use 'enum class' (C++11) instead of traditional enums\n";
    cout << "2. Enum classes are type-safe and don't pollute namespace\n";
    cout << "3. You can specify underlying type (uint8_t, int, etc.)\n";
    cout << "4. Use switch statements for enum-based logic\n";
    cout << "5. Explicit cast needed to convert enum class to int\n";
    cout << "6. Enums are great for state machines and options\n";
    cout << "================================================\n";

    return 0;
}
