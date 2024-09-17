#include "Perceptron.cpp"
#include "TrainingData.cpp"

int main()
{
    int numLayers = 4;
    int neuronsPerLayer[4] = { 2, 2, 2, 1 };

    Perceptron::Perceptron perceptron(numLayers, neuronsPerLayer, 0.2);

    perceptron.initializeWeightsAndBiases();

    std::cout << "Training begin." << std::endl;

    for (int reps = 0; reps < 100000; reps++)
    {
        for (int i = 0; i < NUM_TRAINING_DATA; i++)
        {
            perceptron.backPropogation(trainingInput[i], trainingGoal[i]);
        }
    }

    std::cout << "Training complete." << std::endl;


    float input[2] = { 0, 0 };

    float * output = perceptron.forwardPropogation(input);
    std::cout << "Input = { " << input[0] << ", " << input[1] << " } Output = " << output[0] << std::endl;

    input[0] = 0;
    input[1] = 1;

    output = perceptron.forwardPropogation(input);
    std::cout << "Input = { " << input[0] << ", " << input[1] << " } Output = " << output[0] << std::endl;

    input[0] = 1;
    input[1] = 0;

    output = perceptron.forwardPropogation(input);
    std::cout << "Input = { " << input[0] << ", " << input[1] << " } Output = " << output[0] << std::endl;

    input[0] = 1;
    input[1] = 1;

    output = perceptron.forwardPropogation(input);
    std::cout << "Input = { " << input[0] << ", " << input[1] << " } Output = " << output[0] << std::endl;
    
    return 0;
}