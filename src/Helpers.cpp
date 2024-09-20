
#include "Helpers.hpp"


float Perceptron::dotProduct(float * weight, float * activation, int size)
{
    float sum = 0;

    for (int idx = 0; idx < size; idx++)
    {
        sum += weight[idx] * activation[idx];
    }

    return sum;
}

float Perceptron::sum(float * input, int size)
{
    float sum = 0;

    for (int idx = 0; idx < size; idx++)
    {
        sum += input[idx];
    }

    return sum;
}