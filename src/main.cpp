#include <iostream>

#include <Eigen/Dense>
#include <SFML/Graphics.hpp>

#include "perceptron.hpp"

double relu(double val) {
    return val > 0 ? val : 0;
}

int main() {
    auto window = sf::RenderWindow({600u, 400u}, "CMake SFML");
    window.setFramerateLimit(144);

    // Teach perceptron to add two numbers
    Eigen::MatrixXd inputs(100, 2);
    Eigen::VectorXd results(100);

    int index = 0;
    for(int x = 0; x < 10; x++) {
        for(int y = 0; y < 10; y++) {
            inputs(index, 0) = x;
            inputs(index, 1) = y;
            results(index) = x + y;
            index++;
        }
    }
    
    Eigen::VectorXd final_test(2);
    final_test << 25.0, 25.0;

    Perceptron p(relu, 2); 
    std::cout << "Prediction before training is: " << p.predict(final_test) << std::endl;
    p.train(inputs, results);

    std::cout << "Prediction after training is: " << p.predict(final_test) << std::endl;

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