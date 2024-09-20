
#include "Perceptron.hpp"
#include "../MNIST_for_C/mnist.h"

#define NUM_LAYERS 4
#define TRAINING_EPOCHS 10
#define NUM_CLASSES 10


int main()
{
    // Read MNIST
    load_mnist();


    // Define neural net
    int neuronsPerLayer[NUM_LAYERS] = { 784, 100, 50, 10 };
    Perceptron::Perceptron perceptron(NUM_LAYERS, neuronsPerLayer, 0.01);
    perceptron.initializeWeightsAndBiases();


    // Transform training data to float format.
    float * * transformedTrainingData = new float * [NUM_TRAIN];
    float * * trainingLabels = new float * [NUM_TRAIN];
    for (int i = 0; i < NUM_TRAIN; i++)
    {
        transformedTrainingData[i] = new float[SIZE];
        trainingLabels[i] = new float[NUM_CLASSES];

        trainingLabels[i][train_label_char[i][0]] = 1.0;

        for (int f = 0; f < SIZE; f++)
        {
            transformedTrainingData[i][f] = (float)train_image_char[i][f] / MAX_BRIGHTNESS;
        }
    }


    // Train MLPANN.
    std::cout << "Training begin." << std::endl << std::endl;
    for (int reps = 0; reps < TRAINING_EPOCHS; reps++)
    {
        std::cout << "Epoch " << reps << " begin..." << std::endl;

        for (int i = 0; i < NUM_TRAIN; i++)
        {
            perceptron.backPropogation(transformedTrainingData[i], trainingLabels[i]);
        }
    }
    std::cout << "Training complete." << std::endl;


    // Transform test data.
    float * * transformedTestData = new float * [NUM_TEST];
    for (int i = 0; i < NUM_TEST; i++)
    {
        transformedTestData[i] = new float[SIZE];

        for (int f = 0; f < SIZE; f++)
        {
            transformedTestData[i][f] = (float)test_image_char[i][f] / MAX_BRIGHTNESS;
        }
    }


    // Run forward propagation on test examples.
    int numRight = 0;
    for (int test = 0; test < NUM_TEST; test++)
    {
        float * output = perceptron.forwardPropogation(transformedTestData[test]);
        std::cout << "Input label = " << (int)test_label_char[test][0] << ", Predicted Class = ";

        float highestProb = -1.0;
        int highestClass = -1;
        for (int classNum = 0; classNum < NUM_CLASSES; classNum++)
        {
            if (output[classNum] > highestProb)
            {
                highestClass = classNum;
                highestProb = output[classNum];
            }
        }
        std::cout << highestClass << std::endl;

        if (highestClass == (int)test_label_char[test][0])
        {
            numRight++;
        }
    }
    std::cout << std::endl << "Percentage right: " << ((float)numRight / NUM_TEST) << std::endl;
    
    return 0;
}