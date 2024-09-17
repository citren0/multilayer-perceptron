
#include <cmath>
#include <iostream>
#include "Helpers.cpp"


namespace Perceptron
{
    class Perceptron
    {
        private:
            int numLayers;
            float learningRate;
            int * layerSizes; // layerSizes[layer]
            float * * * weights; // weights[source layer][destination neuron][source neuron]
            float * * biases;
            float * * activations; // activations[layer][neuron]
            float * * z;
        
        public:
            Perceptron(int numLayers, int * layerSizes, float learningRate)
            {
                this->numLayers = numLayers;
                this->layerSizes = layerSizes;
                this->learningRate = learningRate;

                weights = new float * * [numLayers - 1]; // No weights from last layer.
                for (int layer = 0; layer < numLayers - 1; layer++)
                {
                    weights[layer] = new float * [layerSizes[layer + 1]]; // Destination neuron array.

                    for (int source = 0; source < layerSizes[layer]; source++)
                    {
                        weights[layer][source] = new float[layerSizes[layer]]; // Source neuron array.
                    }
                }

                activations = new float * [numLayers];
                biases = new float * [numLayers];
                z = new float * [numLayers];
                for (int layer = 0; layer < numLayers; layer++)
                {
                    activations[layer] = new float[layerSizes[layer]];
                    z[layer] = new float[layerSizes[layer]];
                    biases[layer] = new float[layerSizes[layer]];
                }

            }


            ~Perceptron()
            {
                for (int layer = 0; layer < numLayers - 1; layer++)
                {

                    for (int source = 0; source < layerSizes[layer]; source++)
                    {
                        delete[] weights[layer][source];
                    }

                    delete[] weights[layer];
                }

                delete[] weights;

                for (int layer = 0; layer < numLayers; layer++)
                {
                    delete[] activations[layer];
                    delete[] z[layer];
                    delete[] biases[layer];
                }

                delete[] activations;
                delete[] biases;
                delete[] z;
            }


            void print()
            {
                std::cout << "Weights:" << std::endl;
                for (int layer = 0; layer < numLayers - 1; layer++)
                {
                    std::cout << "Layer " << layer << " weights:";
                    for (int destNeuron = 0; destNeuron < layerSizes[layer + 1]; destNeuron++)
                    {
                        for (int sourceNeuron = 0; sourceNeuron < layerSizes[layer]; sourceNeuron++)
                        {
                            std::cout << sourceNeuron << " -> " << destNeuron << ": " << weights[layer][destNeuron][sourceNeuron] << std::endl;
                        }
                    }
                    std::cout << std::endl << std::endl;
                }

                std::cout << "Biases" << std::endl;
                for (int layer = 0; layer < numLayers; layer++)
                {
                    for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
                    {
                        std::cout << biases[layer][neuron] << " ";
                    }
                    std::cout << std::endl;
                }

                std::cout << "Activations" << std::endl;
                for (int layer = 0; layer < numLayers; layer++)
                {
                    for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
                    {
                        std::cout << activations[layer][neuron] << " ";
                    }
                    std::cout << std::endl;
                }
            }


            void initializeWeightsAndBiases()
            {
                srand(time(NULL));

                for (int layer = 0; layer < numLayers - 1; layer++)
                {
                    for (int destNeuron = 0; destNeuron < layerSizes[layer + 1]; destNeuron++)
                    {
                        for (int sourceNeuron = 0; sourceNeuron < layerSizes[layer]; sourceNeuron++)
                        {
                            weights[layer][destNeuron][sourceNeuron] = 2 * ((float) rand()) / (float) RAND_MAX;
                            weights[layer][destNeuron][sourceNeuron] -= 1;
                        }
                    }
                }

                for (int layer = 0; layer < numLayers; layer++)
                {
                    for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
                    {
                        biases[layer][neuron] = 2 * ((float) rand()) / (float) RAND_MAX;
                        biases[layer][neuron] -= 1;
                    }
                }
            }


            float * forwardPropogation(float * input)
            {
                int outputSize = layerSizes[numLayers - 1];

                // For each layer
                for (int layer = 0; layer < numLayers; layer++)
                {
                    // For each neuron in that layer.
                    for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
                    {
                        if (layer == 0)
                        {
                            // Activation equals input.
                            z[layer][neuron] = input[neuron] + biases[layer][neuron];

                            activations[layer][neuron] = input[neuron] + biases[layer][neuron];
                        }
                        else
                        {
                            z[layer][neuron] = dotProduct(
                                                            weights[layer - 1][neuron],
                                                            activations[layer - 1],
                                                            layerSizes[layer - 1]
                                                         ) + biases[layer][neuron];

                            activations[layer][neuron] = sigmoid(z[layer][neuron], false);
                        }
                    }
                }

                return activations[numLayers - 1];
            }


            void backPropogation(float * input, float * goal)
            {
                // Forward prop
                float * forwardResult = forwardPropogation(input);

                float * * errors = new float * [numLayers];

                for (int layer = 0; layer < numLayers; layer++)
                {
                    errors[layer] = new float[layerSizes[layer]];
                }

                // Output layer errors.
                for (int neuron = 0; neuron < layerSizes[numLayers - 1]; neuron++)
                {
                    errors[numLayers - 1][neuron] = cost(goal[neuron], forwardResult[neuron]);
                }

                // Hidden layers errors.
                for (int layer = numLayers - 2; layer >= 0; layer--)
                {
                    for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
                    {
                        float errorSum = 0.0;

                        for (int next = 0; next < layerSizes[layer + 1]; next++)
                        {
                            errorSum += (errors[layer + 1][next] * weights[layer][next][neuron]);
                        }
                        
                        errors[layer][neuron] = errorSum * sigmoid(z[layer][neuron], true);
                    }
                }

                // Adjust Weights.
                for (int layer = 0; layer < numLayers - 1; layer++)
                {
                    for (int dest = 0; dest < layerSizes[layer + 1]; dest++)
                    {
                        for (int source = 0; source < layerSizes[layer]; source++)
                        {
                            weights[layer][dest][source] -= learningRate * (errors[layer + 1][dest] * activations[layer][source]);
                        }
                    }
                }

                // Adjust Biases.
                for (int layer = 0; layer < numLayers; layer++)
                {
                    for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
                    {
                        biases[layer][neuron] -= learningRate * (errors[layer][neuron]);
                    }
                }

                // Free all memory.
                for (int layer = 0; layer < numLayers; layer++)
                    delete[] errors[layer];
                delete[] errors;
            }


            float sigmoid(float x, bool derivative)
            {
                if (derivative)
                {
                    return sigmoid(x, false) * (1 - sigmoid(x, false));
                }
                else
                {
                    return 1.0 / (1.0 + exp(-x));
                }
            }

            float cost(float expected, float calculated)
            {
                return (calculated - expected);
            }

    };

};