
#include "Perceptron.hpp"
#include "Helpers.hpp"


namespace Perceptron
{
    Perceptron::Perceptron(int numLayers, int * layerSizes, float learningRate)
    {
        this->numLayers = numLayers;
        this->layerSizes = layerSizes;
        this->learningRate = learningRate;

        layerOffsets = new int[numLayers];
        weightLayerOffsets = new int[numLayers];

        // Calculate totalNumNeurons.
        int sum = 0;
        for (int layer = 0; layer < numLayers; layer++)
        {
            // Layer offset is the partial sum up to layer - 1.
            layerOffsets[layer] = sum;

            sum += layerSizes[layer];
        }
        totalNumNeurons = sum;

        // Calculate weightLayerOffsets.
        int weightsSoFar = 0;
        for (int layer = 0; layer < numLayers - 1; layer++)
        {
            weightLayerOffsets[layer] = weightsSoFar;

            weightsSoFar += (layerSizes[layer] * layerSizes[layer + 1]);
        }

        // Weights array size will be the total weightLayerOffsets
        weights = new float[weightsSoFar];

        // Biases, activations, and z will be the size of the total number of neurons.
        biases = new float[totalNumNeurons];
        activations = new float[totalNumNeurons];
        z = new float[totalNumNeurons];

    }


    Perceptron::~Perceptron()
    {
        delete[] layerOffsets;
        delete[] weightLayerOffsets;
        delete[] weights;
        delete[] biases;
        delete[] activations;
        delete[] z;
    }


    void Perceptron::print()
    {
        std::cout << "Weights:" << std::endl;
        for (int layer = 0; layer < numLayers - 1; layer++)
        {
            std::cout << "Layer " << layer << " weights:";
            for (int destNeuron = 0; destNeuron < layerSizes[layer + 1]; destNeuron++)
            {
                for (int sourceNeuron = 0; sourceNeuron < layerSizes[layer]; sourceNeuron++)
                {
                    std::cout << sourceNeuron << " -> " << destNeuron << ": " << weights[getWeightIdx(layer, destNeuron, sourceNeuron)] << std::endl;
                }
            }
            std::cout << std::endl << std::endl;
        }

        std::cout << "Biases" << std::endl;
        for (int layer = 0; layer < numLayers; layer++)
        {
            for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
            {
                std::cout << biases[layerOffsets[layer] + neuron] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Activations" << std::endl;
        for (int layer = 0; layer < numLayers; layer++)
        {
            for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
            {
                std::cout << activations[layerOffsets[layer] + neuron] << " ";
            }
            std::cout << std::endl;
        }
    }


    void Perceptron::printActivations()
    {
        std::cout << "Activations" << std::endl;
        for (int layer = 0; layer < numLayers; layer++)
        {
            for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
            {
                std::cout << activations[layerOffsets[layer] + neuron] << " ";
            }
            std::cout << std::endl;
        }
    }


    void Perceptron::initializeWeightsAndBiases()
    {
        srand(time(NULL));

        for (int layer = 0; layer < numLayers - 1; layer++)
        {
            for (int destNeuron = 0; destNeuron < layerSizes[layer + 1]; destNeuron++)
            {
                for (int sourceNeuron = 0; sourceNeuron < layerSizes[layer]; sourceNeuron++)
                {
                    weights[getWeightIdx(layer, destNeuron, sourceNeuron)] = 2 * ((float) rand()) / (float) RAND_MAX;
                    weights[getWeightIdx(layer, destNeuron, sourceNeuron)] -= 1;
                }
            }
        }

        for (int layer = 0; layer < numLayers; layer++)
        {
            for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
            {
                biases[layerOffsets[layer] + neuron] = 2 * ((float) rand()) / (float) RAND_MAX;
                biases[layerOffsets[layer] + neuron] -= 1;
            }
        }
    }


    float * Perceptron::forwardPropagation(float * input)
    {
        // For each layer
        for (int layer = 0; layer < numLayers; layer++)
        {
            // For each neuron in that layer.
            for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
            {
                if (layer == 0)
                {
                    // Activation equals input.
                    z[neuron] = input[neuron] + biases[neuron];

                    activations[neuron] = input[neuron] + biases[neuron];
                }
                else
                {
                    z[layerOffsets[layer] + neuron] = dotProduct(
                                                        &(weights[getWeightIdx(layer - 1, neuron, 0)]),
                                                        &activations[layerOffsets[layer - 1]],
                                                        layerSizes[layer - 1]
                                                      ) + biases[layerOffsets[layer] + neuron];

                    activations[layerOffsets[layer] + neuron] = sigmoid(z[layerOffsets[layer] + neuron], false);
                }
            }
        }

        // Do not free result of this function.
        return &(activations[layerOffsets[numLayers - 1]]);
    }


    void Perceptron::backPropagation(float * input, float * goal)
    {
        // Forward prop
        float * forwardResult = forwardPropagation(input);

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
                    errorSum += (errors[layer + 1][next] * weights[getWeightIdx(layer, next, neuron)]);
                }
                
                errors[layer][neuron] = errorSum * sigmoid(z[layerOffsets[layer] + neuron], true);
            }
        }

        // Adjust Weights.
        for (int layer = 0; layer < numLayers - 1; layer++)
        {
            for (int dest = 0; dest < layerSizes[layer + 1]; dest++)
            {
                for (int source = 0; source < layerSizes[layer]; source++)
                {
                    weights[getWeightIdx(layer, dest, source)] -= learningRate * (errors[layer + 1][dest] * activations[layerOffsets[layer] + source]);
                }
            }
        }

        // Adjust Biases.
        for (int layer = 0; layer < numLayers; layer++)
        {
            for (int neuron = 0; neuron < layerSizes[layer]; neuron++)
            {
                biases[layerOffsets[layer] + neuron] -= learningRate * (errors[layer][neuron]);
            }
        }

        // Free all memory.
        for (int layer = 0; layer < numLayers; layer++)
            delete[] errors[layer];
        delete[] errors;
    }


    float Perceptron::sigmoid(float x, bool derivative)
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

    float Perceptron::cost(float expected, float calculated)
    {
        return (calculated - expected);
    }


    int Perceptron::getWeightIdx(int layer, int dest, int src)
    {
        return (weightLayerOffsets[layer] + (dest * layerSizes[layer]) + src);
    }

};