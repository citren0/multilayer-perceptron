
#include <cmath>
#include <cmath>
#include <iostream>


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
            Perceptron(int numLayers, int * layerSizes, float learningRate);
            ~Perceptron();
            void print();
            void printActivations();
            void initializeWeightsAndBiases();
            float * forwardPropogation(float * input);
            void backPropogation(float * input, float * goal);
            float sigmoid(float x, bool derivative);
            float cost(float expected, float calculated);

    };

};