#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <Eigen/Dense>

#include <cmath>
#include <functional>
#include <vector>

class Perceptron {
private:
    int input_size;

    double learning_rate;
    double bias;

    Eigen::VectorXd weights;    

    std::function<double(double)> activation;

public:
    Perceptron(std::function<double(double)> activation, unsigned int input_size, double bias = 0.0, double learning_rate = 1.0);

    void updateBias(const double value);
    void train(Eigen::MatrixXd inputs, Eigen::VectorXd expected);

    double predict(Eigen::VectorXd inputs);
    double partialLoss(Eigen::VectorXd inputs, double expected);
};

Perceptron::Perceptron(std::function<double(double)> activation, unsigned int input_size, double bias, double learning_rate) {
    this->activation = activation;
    this->input_size = input_size;
    this->learning_rate = learning_rate;

    // Start with random weights
    this->weights = Eigen::VectorXd::Random(this->input_size);
}


void Perceptron::updateBias(const double diff) {
    this->bias += diff * learning_rate;
}

void Perceptron::train(Eigen::MatrixXd inputs, Eigen::VectorXd expected) {
    double loss = 0.0;
    
    for(unsigned int i = 0; i < expected.size(); i++) {
        loss += this->partialLoss(inputs.row(i), expected[i]);
    }
    loss /= expected.size();

    std::cout << "Total loss is: " << loss << std::endl;
}

double Perceptron::predict(Eigen::VectorXd inputs) {
    return this->activation(inputs.dot(this->weights) + this->bias);
}

double Perceptron::partialLoss(Eigen::VectorXd inputs, double expected) {
    double yh = this->predict(inputs);

    // Clamp to avoid log being nan or inf
    yh = std::clamp(yh, 1e-10, 1.0 - 1e-10);

    // Cross Entropy Loss
    return  -(expected * std::log10(yh) + (1.0 - expected) * std::log10(1.0 - yh));
}

#endif