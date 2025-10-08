import joblib
import os


def generate_C_regression(coefs: list, intercept: float, features: list) -> str:
    # prepare thetas array literal (intercept first)
    thetas = [intercept] + coefs
    thetas_c = ", ".join(f"{v:.8f}f" for v in thetas)

    prediction_func = f"""
float prediction(float* features, int n_features)
{{
    // thetas: [intercept, coef1, coef2, ...]
    float thetas[{len(thetas)}] = {{{thetas_c}}};
    float pred = thetas[0];
    for (int i = 0; i < n_features; i++) {{
        pred += features[i] * thetas[i + 1];
    }}
    return pred;
}}
"""

    sample_c = ", ".join(f"{v:.8f}f" for v in features)

    main_func = f"""
int main(void)
{{
    float features[{len(features)}] = {{{sample_c}}};
    float result = prediction(features, {len(features)});
    printf("Prediction: %f\\n", result);
    return 0;
}}
"""

    c_code = f"#include <stdio.h>\n\n{prediction_func}\n{main_func}\n"
    return c_code


def main():
    model_path = "data/regression.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    coefs, intercept_val = list(model.coef_), float(model.intercept_)

    # create a sample static feature array for main
    features = [205.9991686803, 2, 0]
    print(f"Model coefficients: {coefs}")
    print(f"Using features: {features}")

    # Make C code
    c_code = generate_C_regression(coefs, intercept_val, features)
    out_c = "generated_prediction.c"

    with open(out_c, "w") as f:
        f.write(c_code)

    # Print instructions
    compile_cmd = f"gcc -O2 {out_c} -o generated_prediction && ./generated_prediction"
    print(f"\nWrote C code to: {out_c}")
    print("To compile and run, you can run:")
    print(compile_cmd)


if __name__ == "__main__":
    main()
