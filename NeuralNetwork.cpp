#include "NeuralNetwork.h"

#include <ctime>

NeuralNetwork::NeuralNetwork(std::vector<uint32_t> layers_lengths, func_ptr activations, func_ptr activations_derivatives)
	: activations(activations), activations_derivatives(activations_derivatives)
{
	if (layers_lengths.size() < 2)
		throw new std::exception("The NN must has at least two layers for inputs and outputs");

	for (uint32_t length : layers_lengths)
		if (length)
			neurons.push_back(std::vector<double>(length));
		else
			throw new std::exception("Each layer must have at least one neuron");

	weights.resize(layers_lengths.size() - 1);
	srand(std::time(0));
	for (size_t i = 0; i < layers_lengths.size() - 1; i++)
	{
		weights[i].resize(layers_lengths[i + 1]);

		for (size_t ii = 0; ii < layers_lengths[i + 1]; ii++)
		{
			weights[i][ii].resize(layers_lengths[i] + 1/*for bias*/);

			for (size_t iii = 0; iii < weights[i][ii].size() - 1/*initially, bias = 0.f*/; iii++)
			{
				//give the rest of the weights random value betweem -0.1 and 0.1
				weights[i][ii][iii] = rand() / (double)RAND_MAX * 0.2 - 0.1;
			}
		}
	}
}

void NeuralNetwork::forwad_pass(std::vector<double> inputs)
{
	if (neurons[0].size() != inputs.size())
		throw new std::exception("Invalid inputs count");

	neurons[0] = inputs;

	for (size_t layer = 1; layer < neurons.size(); layer++)
	{
		for (size_t neuron = 0; neuron < neurons[layer].size(); neuron++)
		{
			//multiply the current neuron's inputs with its weights, and add the products
			neurons[layer][neuron] = weights[layer - 1][neuron].back();//bias
			for (size_t i = 0; i < neurons[layer - 1].size(); i++)
				neurons[layer][neuron] += neurons[layer - 1][i] * weights[layer - 1][neuron][i];
			
			//apply the activation function to the neuron value
			neurons[layer][neuron] = activations[layer-1](neurons[layer][neuron]);
		}
	}
}

void NeuralNetwork::backward_pass(std::vector<double> desired_outputs, double learning_rate)
{
	//this will hold all the neurons' error functions derivatives
	std::vector<std::vector<double>> errors(weights.size());

	//calculate the output neorons' error functions derivatives
	for (size_t output = 0; output < neurons.back().size(); output++)
		errors.back().push_back((neurons.back()[output] - desired_outputs[output]) * activations_derivatives.back()(neurons.back()[output]));

	//calculate the hidden neorons' error functions derivatives
	for (size_t layer = neurons.size() - 2; layer > 0; layer--)
	{
		for (size_t neuron = 0; neuron < neurons[layer].size(); neuron++)
		{
			double weighted_errors = 0.0;
			for (size_t i = 0; i < errors[layer].size(); i++)
				weighted_errors += errors[layer][i] * weights[layer][i][neuron];
			errors[layer - 1].push_back(weighted_errors * activations_derivatives[layer - 1](neurons[layer][neuron]));
		}
	}

	//tune the weights of all neurons
	for (size_t i = 0; i < weights.size(); i++)
		for (size_t ii = 0; ii < weights[i].size(); ii++)
		{
			for (size_t iii = 0; iii < weights[i][ii].size() - 1; iii++)
				weights[i][ii][iii] -= learning_rate * errors[i][ii] * neurons[i][iii];
			weights[i][ii].back() -= learning_rate * errors[i].back();
		}
}
