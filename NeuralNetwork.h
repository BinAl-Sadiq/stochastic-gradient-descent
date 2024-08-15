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

public:
	NeuralNetwork() = delete;
	NeuralNetwork(std::vector<uint32_t> layers_lengths, func_ptr activations, func_ptr activations_derivatives);

public:
	void forward_pass(std::vector<double> inputs);
	void backward_pass(std::vector<double> desired_outputs, double learning_rate);
};
