// Example 3: Multiple Namespaces with Same Function Names
// This demonstrates how namespaces prevent naming conflicts

#include <iostream>

// Graphics library namespace
namespace Graphics {
    void draw() {
        std::cout << "Drawing shapes on screen" << std::endl;
    }

    void render() {
        std::cout << "Rendering graphics" << std::endl;
    }
}


// Audio library namespace
namespace Audio {
    void draw() {
        std::cout << "Drawing audio waveform" << std::endl;
    }

    void play() {
        std::cout << "Playing sound" << std::endl;
    }
}

// Game namespace using both libraries
namespace Game {
    void draw() {
        std::cout << "Drawing game scene" << std::endl;
    }
}

int main() {
    // Same function name, but different namespaces - no conflict!
    Graphics::draw();
    Audio::draw();
    Game::draw();

    Graphics::render();
    Audio::play();

    return 0;
}
