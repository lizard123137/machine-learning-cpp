#include <iostream>

#include <Eigen/Dense>
#include <SFML/Graphics.hpp>

#include "perceptron.hpp"

double relu(double val) {
    return val < 0 ? 0 : 1;
}

int main() {
    auto window = sf::RenderWindow({600u, 400u}, "CMake SFML");
    window.setFramerateLimit(144);

    Eigen::MatrixXd inputs(4, 2);
    inputs <<   0.0, 0.0,
                0.0, 1.0,
                1.0, 0.0,
                1.0, 1.0;

    Eigen::VectorXd results(4);
    results << 0.0, 0.0, 0.0, 1.0;

    Perceptron p(relu, 2); 
    p.train(inputs, results);

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