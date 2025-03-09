# stochastic-gradient-descent
Very simple c++ DNN implementation that uses the stochastic gradient descent optimization algorithm

## How to use it?
1. Create an object of type "NeuralNetwork":
   
   ```c++
    //specify the neurons cuont at each layer
	std::vector<uint32_t> layers_lengths = { 250, 40, 30, 5 };

	//specify the activation functions
	NeuralNetwork::acts_t activations = { 3, tanh };
	
	//specify the activation functions' derivatives
	NeuralNetwork::acts_t activations_derivatives = { 3, [](double x) {return 1.0 - x * x; } };

    //specify the criterion
    NeuralNetwork::cri_t criterion = [](std::vector<double> p, std::vector<double> y) {double loss = 0.0; for (int i = 0; i < p.size(); i++) loss += pow(p[i] - y[i], 2); loss /= p.size(); return loss; };

    //specify the criterion's derivative
    NeuralNetwork::cri_d_t criterion_derivative = [](double p, double y) {return p - y; });

	NeuralNetwork NN(layers_lengths, activations, activations_derivatives, criterion, criterion_derivative);
   ```

2. Call the "forward_pass" function to calculate the output layer values:
    ```c++
    //assuming that the vector "inputs" is defined somewhere
    NN.forward_pass(sample);
   ```

3. You can read the output layer values from the member variable "neurons":
   ```c++
   for (size_t i = 0; i < NN.neurons.back().size(); i++)
   {
     double output = NN.neurons.back()[i];
     //...
   }
   ```

4. Calculate the loss
   ```c++
   double loss = NN.loss(desired_output);
   ```

5. To optimize the Neural Network, call the "backward_pass" function
   ```c++
   //the vector "desired_output" holds the correct values that the neural network was supposed to give 
   NN.backward_pass(desired_output, 0.3/*learning rate*/);
   ```

## License
[MIT License](https://github.com/BinAl-Sadiq/stochastic-gradient-descent/blob/main/LICENSE)
