# include <stdio.h>


// https://medium.com/@nsomazr/part-1-implementing-linear-regression-in-c-8d7bb7f603a4
float linear_regression_prediction(float* features, float* thetas, int n_parameters)
{
    float prediction = thetas[0];
    for (int i = 0; i < n_parameters - 1; i++) {
        prediction += features[i] * thetas[i + 1];
    }
    return prediction;
}

// https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
float exp_approx(float x, int n_term)
{
    double sum = 1.0; 
    double term = 1.0; 
    int n = 1; 
 
    while (n <= n_term) {
        term *= x / n;
        sum += term;
        n++;
    }

    return sum;
}

float sigmoid(float x)
{
    return 1 / (1 + exp_approx(-x, 10));
}

// https://mrmint.fr/logistic-regression-machine-learning-introduction-simple
float logistic_regression(float* features, float* thetas, int n_parameter)
{
    float linear_combination = linear_regression_prediction(features, thetas, n_parameter);
    return sigmoid(linear_combination);
}

int simple_tree(float * features, int n_features)
{
    // return 1 - ((features[0] > 0) | (features[1] > 0));
    if (features[0] > 0)
        return 0;
    else
        if (features[1] > 0)
            return 0;
    return 1;
}

int main(){
    
    // TEST linear_regression_prediction
    float X[] = {1, 1, 1};
    float theta[] = {0, 1, 1, 1};
    float result = linear_regression_prediction(X, theta, 4);
    printf("Prediction: %f\n", result);
    
    // TEST exp_approx
    float app = exp_approx(1, 5);
    printf("Exponential approximation at e^1: %f\n", app);

    return 0;
}
