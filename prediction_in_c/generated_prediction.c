#include <stdio.h>


float prediction(float* features, int n_features)
{
    // thetas: [intercept, coef1, coef2, ...]
    float thetas[4] = {-8152.93771016f, 717.25836971f, 36824.19597426f, 101571.84002157f};
    float pred = thetas[0];
    for (int i = 0; i < n_features; i++) {
        pred += features[i] * thetas[i + 1];
    }
    return pred;
}


int main(void)
{
    float features[3] = {205.99916868f, 2.00000000f, 0.00000000f};
    float result = prediction(features, 3);
    printf("Prediction: %f\n", result);
    return 0;
}

