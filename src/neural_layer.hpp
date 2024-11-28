#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include <Eigen/Dense>

#include <functional>
#include <vector>

#include "perceptron.hpp"

class NeuralLayer {
private:
    std::vector<Perceptron> perceptrons;
public:
    NeuralLayer(unsigned int size, std::function<double(double)> activation, unsigned int input_size, double bias = 0.0, double learning_rate = 0.01);
    
    unsigned int getLayerSize() const;
    Eigen::VectorXd getPredictions(Eigen::VectorXd inputs) const;


};

NeuralLayer::NeuralLayer(unsigned int size, std::function<double(double)> activation, unsigned int input_size, double bias, double learning_rate) {
    for(unsigned int i = 0; i < size; i++) {
        this->perceptrons.push_back(Perceptron(activation, input_size, bias, learning_rate));
    }
}

unsigned int NeuralLayer::getLayerSize() const {
    return this->perceptrons.size();
}

Eigen::VectorXd NeuralLayer::getPredictions(Eigen::VectorXd inputs) const {
    Eigen::VectorXd result(this->perceptrons.size());

    unsigned int idx = 0;
    for (auto& p : this->perceptrons) {
        result(idx) = p.predict(inputs);
    }

    return result;
}

#endif 