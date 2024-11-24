#include <SFML/Graphics.hpp>

int main() {
    auto window = sf::RenderWindow({600u, 400u}, "CMake SFML");
    window.setFramerateLimit(144);

    while (window.isOpen()) {
        for (auto event = sf::Event(); window.pollEvent(event);) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        window.clear();
        window.display();
    }

}