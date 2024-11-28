#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <Eigen/Dense>

#include <vector>
#include <string>
#include <sstream>

#include "perceptron.hpp"

class Serializer {
private:
    Serializer();

public:
    static std::string SerializePerceptronJSON(const Perceptron& perceptron);
};

std::string Serializer::SerializePerceptronJSON(const Perceptron& perceptron) {
    // TODO switch to nlohmann library for json support
    std::vector<double> weights;
    weights.resize(perceptron.weights.size());
    Eigen::VectorXd::Map(&weights[0], perceptron.weights.size()) = perceptron.weights;

    std::stringstream ss;
    ss << "\"perceptron\": {\n"
        << "\t\"input_size\": " << perceptron.input_size << ",\n"
        << "\t\"learning_rate\": " << perceptron.learning_rate << ",\n"
        << "\t\"bias\": " << perceptron.bias << ",\n"
        << "\t\"weights\": [";

    bool first = true;
    for (auto v : weights) {
        ss << v;
        if (first)
            ss << ", ";
        first = false;
    }
        
    ss << "],\n"
        << "}\n";
    return ss.str();
}

#endif