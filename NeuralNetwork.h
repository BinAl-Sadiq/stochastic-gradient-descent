#pragma once

#include <vector>

class NeuralNetwork
{
public:
	typedef std::vector<double(*)(double)> func_ptr;

public:
	std::vector<std::vector<double>> neurons;
	std::vector<std::vector<std::vector<double>>> weights;
	
	func_ptr activations;
	func_ptr activations_derivatives;
	double(*criterion)(std::vector<double> p, std::vector<double> y);
	double(*criterion_derivative)(double p, double y);

public:
	NeuralNetwork() = delete;
	NeuralNetwork(std::vector<uint32_t> layers_lengths, func_ptr acts, func_ptr acts_d, double(*c)(std::vector<double> p, std::vector<double> y), double(*c_d)(double p, double y));

public:
	void forward_pass(std::vector<double> inputs);
	double loss(std::vector<double> desired_outputs);
	void backward_pass(std::vector<double> desired_outputs, double learning_rate);
};
