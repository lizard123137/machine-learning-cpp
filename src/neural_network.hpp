#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <Eigen/Dense>
#include <vector>

#include "neural_layer.hpp"

class NeuralNetwork {
private:
    std::vector<NeuralLayer> layers;

public:
    NeuralNetwork();

    void addLayer(NeuralLayer layer);
    Eigen::VectorXd predict(Eigen::VectorXd inputs) const;
};

void NeuralNetwork::addLayer(NeuralLayer layer) {
    this->layers.push_back(layer);
}

Eigen::VectorXd NeuralNetwork::predict(Eigen::VectorXd inputs) const {
    Eigen::VectorXd layerInput = inputs;

    for(auto layer : this->layers) {
        layerInput = layer.getPredictions(layerInput);
    }

    return layerInput;
}

#endif