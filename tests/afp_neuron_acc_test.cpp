#include "../include/afp_encoded_tensor.hpp"
#include "../include/afp_math.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static constexpr int NUM_SINGLE_TESTS = 1000;
static constexpr int NUM_NEURONS = 8;
static constexpr int INPUT_SIZE = 16;

static constexpr float VALUE_MIN = -2.0f;
static constexpr float VALUE_MAX = 2.0f;

static float randomFloat(
    std::mt19937 &rng,
    float min_value,
    float max_value)
{
    std::uniform_real_distribution<float> distribution(
        min_value,
        max_value);

    return distribution(rng);
}

static float fp32DotProduct(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    float result = 0.0f;

    for (std::size_t i = 0; i < a.size(); ++i)
        result += a[i] * b[i];

    return result;
}

static std::vector<float> fp32Layer(
    const std::vector<float> &input,
    const std::vector<std::vector<float>> &weights,
    const std::vector<float> &bias)
{
    std::vector<float> output(weights.size());

    for (std::size_t neuron = 0; neuron < weights.size(); ++neuron)
    {
        output[neuron] =
            fp32DotProduct(input, weights[neuron])
            + bias[neuron];
    }

    return output;
}

static float absoluteError(
    float expected,
    float actual)
{
    return std::fabs(expected - actual);
}

static float relativeError(
    float expected,
    float actual)
{
    const float denominator =
        std::max(std::fabs(expected), 1e-6f);

    return std::fabs(expected - actual) / denominator;
}

static void printVector(
    const std::vector<float> &values)
{
    for (float value : values)
        std::cout << std::setw(12) << value << ' ';

    std::cout << '\n';
}

int main()
{
    std::mt19937 rng(12345);

    /*
     * Change this configuration if your AFPConfig
     * requires specific settings.
     */
    AFPConfig config;

    AFPQuantizer quantizer(config);

    /*
     * ============================================================
     * SINGLE NEURON TEST
     * ============================================================
     */

    std::cout << "========================================\n";
    std::cout << "Single Neuron Random Accuracy Test\n";
    std::cout << "========================================\n";

    float total_absolute_error = 0.0f;
    float total_relative_error = 0.0f;

    float maximum_absolute_error = 0.0f;
    float maximum_relative_error = 0.0f;

    int passed = 0;

    for (int test = 0; test < NUM_SINGLE_TESTS; ++test)
    {
        std::vector<float> input(INPUT_SIZE);
        std::vector<float> weights(INPUT_SIZE);

        for (int i = 0; i < INPUT_SIZE; ++i)
        {
            input[i] =
                randomFloat(rng, VALUE_MIN, VALUE_MAX);

            weights[i] =
                randomFloat(rng, VALUE_MIN, VALUE_MAX);
        }

        const float expected =
            fp32DotProduct(input, weights);

        const AFPEncodedTensor afpInput =
            quantizer.encode(input);

        const AFPEncodedTensor afpWeights =
            quantizer.encode(weights);

        AFPEncodedTensor afpResult =
            AFPArithmetic::dotProduct(
                afpInput,
                afpWeights);

        const std::vector<float> decoded =
            quantizer.decode(afpResult);

        const float actual = decoded[0];

        const float abs_error =
            absoluteError(expected, actual);

        const float rel_error =
            relativeError(expected, actual);

        total_absolute_error += abs_error;
        total_relative_error += rel_error;

        if (abs_error > maximum_absolute_error)
            maximum_absolute_error = abs_error;

        if (rel_error > maximum_relative_error)
            maximum_relative_error = rel_error;

        /*
         * There is no strict pass/fail accuracy threshold here.
         *
         * This is primarily a numerical characterization test.
         * We consider a result reasonable if the absolute error
         * is below 0.25 for this initial experiment.
         */
        if (abs_error <= 0.25f)
            ++passed;

        /*
         * Print the first few examples so we can inspect
         * individual neuron behavior.
         */
        if (test < 5)
        {
            std::cout << "\nTest " << test + 1 << '\n';
            std::cout << "Expected: " << expected << '\n';
            std::cout << "AFP:      " << actual << '\n';
            std::cout << "Abs err:  " << abs_error << '\n';
            std::cout << "Rel err:  " << rel_error << '\n';
        }
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << "Single Neuron Results\n";
    std::cout << "----------------------------------------\n";

    std::cout << std::fixed << std::setprecision(6);

    std::cout
        << "Tests:                 "
        << NUM_SINGLE_TESTS
        << '\n';

    std::cout
        << "Passed (< 0.25 error): "
        << passed
        << " / "
        << NUM_SINGLE_TESTS
        << '\n';

    std::cout
        << "Pass rate:             "
        << (100.0f * passed / NUM_SINGLE_TESTS)
        << "%\n";

    std::cout
        << "Mean absolute error:   "
        << (total_absolute_error / NUM_SINGLE_TESTS)
        << '\n';

    std::cout
        << "Mean relative error:   "
        << (total_relative_error / NUM_SINGLE_TESTS)
        << '\n';

    std::cout
        << "Maximum absolute error:"
        << ' '
        << maximum_absolute_error
        << '\n';

    std::cout
        << "Maximum relative error:"
        << ' '
        << maximum_relative_error
        << '\n';


    /*
     * ============================================================
     * MULTI-NEURON TEST
     * ============================================================
     */

    std::cout << "\n\n";
    std::cout << "========================================\n";
    std::cout << "Multi-Neuron Random Accuracy Test\n";
    std::cout << "========================================\n";

    std::vector<float> input(INPUT_SIZE);

    std::vector<std::vector<float>> weights(
        NUM_NEURONS,
        std::vector<float>(INPUT_SIZE));

    std::vector<float> bias(NUM_NEURONS);

    /*
     * Generate one shared input vector.
     */
    for (int i = 0; i < INPUT_SIZE; ++i)
    {
        input[i] =
            randomFloat(rng, VALUE_MIN, VALUE_MAX);
    }

    /*
     * Generate independent weights and biases
     * for every neuron.
     */
    for (int neuron = 0; neuron < NUM_NEURONS; ++neuron)
    {
        for (int i = 0; i < INPUT_SIZE; ++i)
        {
            weights[neuron][i] =
                randomFloat(rng, VALUE_MIN, VALUE_MAX);
        }

        bias[neuron] =
            randomFloat(rng, VALUE_MIN, VALUE_MAX);
    }

    /*
     * Reference FP32 layer.
     */
    const std::vector<float> expected =
        fp32Layer(input, weights, bias);

    /*
     * Encode input once.
     */
    const AFPEncodedTensor afpInput =
        quantizer.encode(input);

    std::vector<float> actual(NUM_NEURONS);

    /*
     * Run every neuron through AFP.
     */
    for (int neuron = 0; neuron < NUM_NEURONS; ++neuron)
    {
        const AFPEncodedTensor afpWeights =
            quantizer.encode(weights[neuron]);

        AFPEncodedTensor afpResult =
            AFPArithmetic::dotProduct(
                afpInput,
                afpWeights);

        const std::vector<float> decoded =
            quantizer.decode(afpResult);

        actual[neuron] =
            decoded[0] + bias[neuron];
    }

    std::cout << "\nFP32 output:\n";
    printVector(expected);

    std::cout << "\nAFP output:\n";
    printVector(actual);

    float layer_total_absolute_error = 0.0f;
    float layer_total_relative_error = 0.0f;

    float layer_max_absolute_error = 0.0f;
    float layer_max_relative_error = 0.0f;

    int layer_passed = 0;

    for (int neuron = 0; neuron < NUM_NEURONS; ++neuron)
    {
        const float abs_error =
            absoluteError(
                expected[neuron],
                actual[neuron]);

        const float rel_error =
            relativeError(
                expected[neuron],
                actual[neuron]);

        layer_total_absolute_error += abs_error;
        layer_total_relative_error += rel_error;

        if (abs_error > layer_max_absolute_error)
            layer_max_absolute_error = abs_error;

        if (rel_error > layer_max_relative_error)
            layer_max_relative_error = rel_error;

        if (abs_error <= 0.25f)
            ++layer_passed;

        std::cout
            << "\nNeuron "
            << neuron
            << ":\n";

        std::cout
            << "  FP32:     "
            << expected[neuron]
            << '\n';

        std::cout
            << "  AFP:      "
            << actual[neuron]
            << '\n';

        std::cout
            << "  Abs err:  "
            << abs_error
            << '\n';

        std::cout
            << "  Rel err:  "
            << rel_error
            << '\n';
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << "Multi-Neuron Results\n";
    std::cout << "----------------------------------------\n";

    std::cout
        << "Neurons:               "
        << NUM_NEURONS
        << '\n';

    std::cout
        << "Passed (< 0.25 error): "
        << layer_passed
        << " / "
        << NUM_NEURONS
        << '\n';

    std::cout
        << "Pass rate:             "
        << (100.0f * layer_passed / NUM_NEURONS)
        << "%\n";

    std::cout
        << "Mean absolute error:   "
        << (layer_total_absolute_error / NUM_NEURONS)
        << '\n';

    std::cout
        << "Mean relative error:   "
        << (layer_total_relative_error / NUM_NEURONS)
        << '\n';

    std::cout
        << "Maximum absolute error:"
        << ' '
        << layer_max_absolute_error
        << '\n';

    std::cout
        << "Maximum relative error:"
        << ' '
        << layer_max_relative_error
        << '\n';

    std::cout << "\n========================================\n";
    std::cout << "Tests complete.\n";
    std::cout << "========================================\n";

    return 0;
}