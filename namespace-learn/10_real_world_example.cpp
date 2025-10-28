// Example 10: Real-World Project Structure
// Demonstrates how namespaces organize a complete application

#include <iostream>
#include <string>
#include <vector>

// Main application namespace
namespace GameEngine {

    // Subsystem for graphics
    namespace Graphics {
        class Renderer {
        public:
            void initialize() {
                std::cout << "[Graphics] Renderer initialized" << std::endl;
            }

            void draw(const std::string& object) {
                std::cout << "[Graphics] Drawing: " << object << std::endl;
            }
        };

        namespace UI {
            void showMenu() {
                std::cout << "[UI] Displaying main menu" << std::endl;
            }
        }
    }

    // Subsystem for physics
    namespace Physics {
        class Engine {
        public:
            void initialize() {
                std::cout << "[Physics] Engine initialized" << std::endl;
            }

            void update(float deltaTime) {
                std::cout << "[Physics] Updating physics (dt=" << deltaTime << ")" << std::endl;
            }
        };
    }

    // Subsystem for audio
    namespace Audio {
        class SoundManager {
        public:
            void initialize() {
                std::cout << "[Audio] Sound manager initialized" << std::endl;
            }

            void playSound(const std::string& soundName) {
                std::cout << "[Audio] Playing: " << soundName << std::endl;
            }
        };
    }

    // Game logic
    namespace Game {
        class World {
        private:
            Graphics::Renderer renderer;
            Physics::Engine physics;
            Audio::SoundManager audio;

        public:
            void initialize() {
                std::cout << "[Game] Initializing game world..." << std::endl;
                renderer.initialize();
                physics.initialize();
                audio.initialize();
                Graphics::UI::showMenu();
            }

            void update(float deltaTime) {
                physics.update(deltaTime);
                renderer.draw("Player");
                renderer.draw("Enemy");
            }

            void playBackgroundMusic() {
                audio.playSound("background_music.mp3");
            }
        };
    }

    // Utilities used across the engine
    namespace Utils {
        void logMessage(const std::string& msg) {
            std::cout << "[LOG] " << msg << std::endl;
        }
    }
}

int main() {
    using namespace GameEngine;

    Utils::logMessage("Starting game engine...");

    Game::World gameWorld;
    gameWorld.initialize();

    Utils::logMessage("Running game loop...");
    gameWorld.update(0.016f);  // 16ms frame time
    gameWorld.playBackgroundMusic();

    Utils::logMessage("Game engine stopped.");

    return 0;
}

// This structure keeps code organized:
// - Graphics, Physics, Audio are separate subsystems
// - Each can be developed independently
// - Clear separation of concerns
// - Easy to find and maintain code
