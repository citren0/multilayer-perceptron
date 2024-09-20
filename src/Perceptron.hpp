
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

            // layerSizes[layer]
            int * layerSizes;

            // totalNumNeurons = sum(layerSizes from 0 to N)
            int totalNumNeurons;
            
            // layerOffsets[layer] = sum(layerSizes from 0 to layer - 1)
            int * layerOffsets;

            // weightLayerOffsets[layer] = sum((layerSizes[layer] * layerSizes[layer + 1]) from 0 to layer - 1)
            int * weightLayerOffsets;

            // weights[(weightLayerOffsets[layer]) + (dest_neuron * layerSize[layer + 1]) + src_neuron]
            float * weights;

            // biases[layerOffsets[layer] + neuron]
            float * biases;

            // activations[layerOffsets[layer] + neuron]
            float * activations;

            // z[layerOffsets[layer] + neuron]
            float * z;
        

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
            int getWeightIdx(int layer, int dest, int src);
    };

};