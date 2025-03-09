#pragma once

#include <vector>

class NeuralNetwork
{
public:
	typedef std::vector<double(*)(double)> acts_t;//type for activation functions and their derivatives
	typedef double(*cri_t)(std::vector<double> p, std::vector<double> y);//type for criterion
	typedef double(*cri_d_t)(double p, double y);//type for criterion's derivative

public:
	std::vector<std::vector<double>> neurons;
	std::vector<std::vector<std::vector<double>>> weights;
	
	acts_t activations;
	acts_t activations_derivatives;
	cri_t criterion;
	cri_d_t criterion_derivative;

public:
	NeuralNetwork() = delete;
	NeuralNetwork(std::vector<uint32_t> layers_lengths, acts_t acts, acts_t acts_d, cri_t c, cri_d_t c_d);

public:
	void forward_pass(std::vector<double> inputs);
	double loss(std::vector<double> desired_outputs);
	void backward_pass(std::vector<double> desired_outputs, double learning_rate);
};
