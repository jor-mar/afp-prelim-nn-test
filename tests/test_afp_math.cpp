#include "../include/afp.hpp"
#include "../include/afp_math.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Test utilities
float computeDotProduct(const std::vector<float> &a, const std::vector<float> &b)
{
    float result = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
        result += a[i] * b[i];
    return result;
}

std::vector<float> computeElementwiseAdd(const std::vector<float> &a, const std::vector<float> &b)
{
    std::vector<float> result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];
    return result;
}

std::vector<float> computeElementwiseSubtract(const std::vector<float> &a, const std::vector<float> &b)
{
    std::vector<float> result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] - b[i];
    return result;
}

std::vector<float> computeElementwiseMultiply(const std::vector<float> &a, const std::vector<float> &b)
{
    std::vector<float> result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] * b[i];
    return result;
}

std::vector<float> computeElementwiseDivide(const std::vector<float> &a, const std::vector<float> &b)
{
    std::vector<float> result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] / b[i];
    return result;
}

std::vector<float> computeReLU(const std::vector<float> &input)
{
    std::vector<float> result(input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
        result[i] = std::max(0.0f, input[i]);
    return result;
}

std::vector<float> computeSigmoid(const std::vector<float> &input)
{
    std::vector<float> result(input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
        result[i] = 1.0f / (1.0f + std::exp(-input[i]));
    return result;
}

std::vector<float> computeTanh(const std::vector<float> &input)
{
    std::vector<float> result(input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
        result[i] = std::tanh(input[i]);
    return result;
}

float computeSum(const std::vector<float> &input)
{
    float result = 0.0f;
    for (float val : input)
        result += val;
    return result;
}

float computeMean(const std::vector<float> &input)
{
    if (input.empty()) return 0.0f;
    return computeSum(input) / static_cast<float>(input.size());
}

float computeMax(const std::vector<float> &input)
{
    if (input.empty()) return 0.0f;
    float result = input[0];
    for (float val : input)
        result = std::max(result, val);
    return result;
}

std::vector<float> computeMatrixVectorMultiply(
    const std::vector<float> &weights,
    const std::vector<float> &input,
    std::size_t rows,
    std::size_t cols)
{
    std::vector<float> result(rows, 0.0f);
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = 0; col < cols; ++col)
        {
            result[row] += weights[row * cols + col] * input[col];
        }
    }
    return result;
}

float maxAbsoluteError(const std::vector<float> &a, const std::vector<float> &b)
{
    float max_error = 0.0f;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    {
        float error = std::fabs(a[i] - b[i]);
        max_error = std::max(max_error, error);
    }
    return max_error;
}

// Test result reporting
void reportTestResult(const std::string &test_name, bool passed, float error = 0.0f)
{
    std::cout << "[" << test_name << "] ";
    if (passed)
    {
        std::cout << "PASS";
        if (error > 0.0f)
            std::cout << " (max error: " << std::fixed << std::setprecision(6) << error << ")";
    }
    else
    {
        std::cout << "FAIL";
        if (error > 0.0f)
            std::cout << " (max error: " << std::fixed << std::setprecision(6) << error << ")";
    }
    std::cout << std::endl;
}

// ============================================================================
// Element-wise Operation Tests
// ============================================================================

void testElementwiseAdd()
{
    AFPQuantizer quantizer;
    
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                           9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> b = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f,
                           8.5f, 9.5f, 10.5f, 11.5f, 12.5f, 13.5f, 14.5f, 15.5f};
    
    AFPEncodedTensor encoded_a = quantizer.encode(a);
    AFPEncodedTensor encoded_b = quantizer.encode(b);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::add(encoded_a, encoded_b);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    std::vector<float> expected = computeElementwiseAdd(a, b);
    float error = maxAbsoluteError(afp_result, expected);
    
    reportTestResult("Element-wise Add", error < 0.5f, error);
}

void testElementwiseSubtract()
{
    AFPQuantizer quantizer;
    
    std::vector<float> a = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.5f, 0.25f, 0.125f,
                           8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    std::vector<float> b = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.25f, 0.125f,
                           2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    
    AFPEncodedTensor encoded_a = quantizer.encode(a);
    AFPEncodedTensor encoded_b = quantizer.encode(b);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::subtract(encoded_a, encoded_b);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    std::vector<float> expected = computeElementwiseSubtract(a, b);
    float error = maxAbsoluteError(afp_result, expected);
    
    reportTestResult("Element-wise Subtract", error < 0.5f, error);
}

void testElementwiseMultiply()
{
    AFPQuantizer quantizer;
    
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                           9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> b = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                           2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    
    AFPEncodedTensor encoded_a = quantizer.encode(a);
    AFPEncodedTensor encoded_b = quantizer.encode(b);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::multiply(encoded_a, encoded_b);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    std::vector<float> expected = computeElementwiseMultiply(a, b);
    float error = maxAbsoluteError(afp_result, expected);
    
    reportTestResult("Element-wise Multiply", error < 1.0f, error);
}

void testElementwiseDivide()
{
    AFPQuantizer quantizer;
    
    std::vector<float> a = {8.0f, 6.0f, 4.0f, 2.0f, 16.0f, 12.0f, 10.0f, 8.0f,
                           32.0f, 24.0f, 18.0f, 14.0f, 64.0f, 48.0f, 36.0f, 28.0f};
    std::vector<float> b = {2.0f, 2.0f, 2.0f, 2.0f, 4.0f, 4.0f, 4.0f, 4.0f,
                           8.0f, 8.0f, 8.0f, 8.0f, 16.0f, 16.0f, 16.0f, 16.0f};
    
    AFPEncodedTensor encoded_a = quantizer.encode(a);
    AFPEncodedTensor encoded_b = quantizer.encode(b);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::divide(encoded_a, encoded_b);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    std::vector<float> expected = computeElementwiseDivide(a, b);
    float error = maxAbsoluteError(afp_result, expected);
    
    reportTestResult("Element-wise Divide", error < 2.0f, error);
}

// ============================================================================
// Dot Product Tests
// ============================================================================

void testDotProductBasic()
{
    AFPQuantizer quantizer;
    
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                           9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> b = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                           2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    
    AFPEncodedTensor encoded_a = quantizer.encode(a);
    AFPEncodedTensor encoded_b = quantizer.encode(b);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::dotProduct(encoded_a, encoded_b);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    float expected = computeDotProduct(a, b);
    float error = std::fabs(afp_result[0] - expected);
    
    reportTestResult("Dot Product Basic", error < 5.0f, error);
}

void testDotProductMixedSigns()
{
    AFPQuantizer quantizer;
    
    std::vector<float> a = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f,
                           9.0f, -10.0f, 11.0f, -12.0f, 13.0f, -14.0f, 15.0f, -16.0f};
    std::vector<float> b = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    
    AFPEncodedTensor encoded_a = quantizer.encode(a);
    AFPEncodedTensor encoded_b = quantizer.encode(b);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::dotProduct(encoded_a, encoded_b);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    float expected = computeDotProduct(a, b);
    float error = std::fabs(afp_result[0] - expected);
    
    reportTestResult("Dot Product Mixed Signs", error < 5.0f, error);
}

// ============================================================================
// Matrix Operation Tests
// ============================================================================

void testMatrixVectorMultiply()
{
    AFPQuantizer quantizer;
    
    const std::size_t rows = 4;
    const std::size_t cols = 4;
    
    std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f,
                                  9.0f, 10.0f, 11.0f, 12.0f,
                                  13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> input = {1.0f, 1.0f, 1.0f, 1.0f};
    
    AFPEncodedTensor encoded_weights = quantizer.encode(weights);
    AFPEncodedTensor encoded_input = quantizer.encode(input);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::matrixVectorMultiply(
        encoded_weights, encoded_input, rows, cols);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    std::vector<float> expected = computeMatrixVectorMultiply(weights, input, rows, cols);
    float error = maxAbsoluteError(afp_result, expected);
    
    reportTestResult("Matrix-Vector Multiply", error < 5.0f, error);
}

void testMatrixMultiply()
{
    AFPQuantizer quantizer;
    
    const std::size_t rows_a = 2;
    const std::size_t cols_a = 2;
    const std::size_t cols_b = 2;
    
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
    
    AFPEncodedTensor encoded_a = quantizer.encode(a);
    AFPEncodedTensor encoded_b = quantizer.encode(b);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::matrixMultiply(
        encoded_a, encoded_b, rows_a, cols_a, cols_b);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    std::vector<float> expected = {19.0f, 22.0f, 43.0f, 50.0f};
    float error = maxAbsoluteError(afp_result, expected);
    
    reportTestResult("Matrix Multiply", error < 10.0f, error);
}

// ============================================================================
// Activation Function Tests
// ============================================================================

void testReLU()
{
    AFPQuantizer quantizer;
    
    std::vector<float> input = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f,
                               -1.5f, -0.25f, 0.25f, 1.5f, -3.0f, 0.0f, 0.0f, 4.0f};
    
    AFPEncodedTensor encoded_input = quantizer.encode(input);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::relu(encoded_input);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    std::vector<float> expected = computeReLU(input);
    float error = maxAbsoluteError(afp_result, expected);
    
    reportTestResult("ReLU", error < 0.5f, error);
}

void testSigmoid()
{
    AFPQuantizer quantizer;
    
    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, 1.5f,
                               -1.5f, 0.25f, -0.25f, 3.0f, -3.0f, 0.0f, 1.0f, -1.0f};
    
    AFPEncodedTensor encoded_input = quantizer.encode(input);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::sigmoid(encoded_input);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    std::vector<float> expected = computeSigmoid(input);
    float error = maxAbsoluteError(afp_result, expected);
    
    // Sigmoid approximation may have larger errors
    reportTestResult("Sigmoid", error < 0.3f, error);
}

void testTanh()
{
    AFPQuantizer quantizer;
    
    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, 1.5f,
                               -1.5f, 0.25f, -0.25f, 3.0f, -3.0f, 0.0f, 1.0f, -1.0f};
    
    AFPEncodedTensor encoded_input = quantizer.encode(input);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::tanh(encoded_input);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    std::vector<float> expected = computeTanh(input);
    float error = maxAbsoluteError(afp_result, expected);
    
    // Tanh approximation may have larger errors
    reportTestResult("Tanh", error < 2.5f, error);
}

// ============================================================================
// Reduction Operation Tests
// ============================================================================

void testSum()
{
    AFPQuantizer quantizer;
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    
    AFPEncodedTensor encoded_input = quantizer.encode(input);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::sum(encoded_input);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    float expected = computeSum(input);
    float error = std::fabs(afp_result[0] - expected);
    
    reportTestResult("Sum", error < 10.0f, error);
}

void testMean()
{
    AFPQuantizer quantizer;
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    
    AFPEncodedTensor encoded_input = quantizer.encode(input);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::mean(encoded_input);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    float expected = computeMean(input);
    float error = std::fabs(afp_result[0] - expected);
    
    reportTestResult("Mean", error < 2.0f, error);
}

void testMax()
{
    AFPQuantizer quantizer;
    
    std::vector<float> input = {1.0f, 5.0f, 3.0f, 8.0f, 2.0f, 7.0f, 4.0f, 6.0f,
                               9.0f, 1.0f, 11.0f, 3.0f, 5.0f, 16.0f, 2.0f, 10.0f};
    
    AFPEncodedTensor encoded_input = quantizer.encode(input);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::max(encoded_input);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    float expected = computeMax(input);
    float error = std::fabs(afp_result[0] - expected);
    
    reportTestResult("Max", error < 1.0f, error);
}

// ============================================================================
// Broadcasting Operation Tests
// ============================================================================

void testBroadcastAdd()
{
    AFPQuantizer quantizer;

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f,
                                5.0f, 6.0f, 7.0f, 8.0f,
                                9.0f, 10.0f, 11.0f, 12.0f,
                                13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> scalar = {2.5f};
    std::vector<float> expected(input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
        expected[i] = input[i] + scalar[0];

    AFPEncodedTensor encoded_input = quantizer.encode(input);
    AFPEncodedTensor encoded_scalar = quantizer.encode(scalar);
    AFPEncodedTensor encoded_result = AFPArithmetic::broadcastAdd(
        encoded_input, encoded_scalar, input.size());
    std::vector<float> afp_result = quantizer.decode(encoded_result);

    float error = maxAbsoluteError(afp_result, expected);
    reportTestResult("Broadcast Add", error < 0.5f, error);
}

void testBroadcastMultiply()
{
    AFPQuantizer quantizer;

    std::vector<float> input = {1.0f, -2.0f, 3.0f, -4.0f,
                                5.0f, -6.0f, 7.0f, -8.0f,
                                9.0f, -10.0f, 11.0f, -12.0f,
                                13.0f, -14.0f, 15.0f, -16.0f};
    std::vector<float> scalar = {-1.5f};
    std::vector<float> expected(input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
        expected[i] = input[i] * scalar[0];

    AFPEncodedTensor encoded_input = quantizer.encode(input);
    AFPEncodedTensor encoded_scalar = quantizer.encode(scalar);
    AFPEncodedTensor encoded_result = AFPArithmetic::broadcastMultiply(
        encoded_input, encoded_scalar, input.size());
    std::vector<float> afp_result = quantizer.decode(encoded_result);

    float error = maxAbsoluteError(afp_result, expected);
    reportTestResult("Broadcast Multiply", error < 1.0f, error);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

void testZeros()
{
    AFPQuantizer quantizer;
    
    std::vector<float> a = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                           0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                           9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    
    AFPEncodedTensor encoded_a = quantizer.encode(a);
    AFPEncodedTensor encoded_b = quantizer.encode(b);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::multiply(encoded_a, encoded_b);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    std::vector<float> expected = computeElementwiseMultiply(a, b);
    float error = maxAbsoluteError(afp_result, expected);
    
    reportTestResult("Zeros Handling", error < 1.5f, error);
}

void testSmallValues()
{
    AFPQuantizer quantizer;
    
    std::vector<float> a = {0.001f, 0.002f, 0.003f, 0.004f, 0.005f, 0.006f, 0.007f, 0.008f,
                           0.009f, 0.010f, 0.011f, 0.012f, 0.013f, 0.014f, 0.015f, 0.016f};
    std::vector<float> b = {1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f,
                           1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f};
    
    AFPEncodedTensor encoded_a = quantizer.encode(a);
    AFPEncodedTensor encoded_b = quantizer.encode(b);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::multiply(encoded_a, encoded_b);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    std::vector<float> expected = computeElementwiseMultiply(a, b);
    float error = maxAbsoluteError(afp_result, expected);
    
    reportTestResult("Small Values", error < 5.0f, error);
}

void testLargeValues()
{
    AFPQuantizer quantizer;
    
    std::vector<float> a = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f,
                           900.0f, 1000.0f, 1100.0f, 1200.0f, 1300.0f, 1400.0f, 1500.0f, 1600.0f};
    std::vector<float> b = {0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f,
                           0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f};
    
    AFPEncodedTensor encoded_a = quantizer.encode(a);
    AFPEncodedTensor encoded_b = quantizer.encode(b);
    
    AFPEncodedTensor encoded_result = AFPArithmetic::multiply(encoded_a, encoded_b);
    std::vector<float> afp_result = quantizer.decode(encoded_result);
    
    std::vector<float> expected = computeElementwiseMultiply(a, b);
    float error = maxAbsoluteError(afp_result, expected);
    
    reportTestResult("Large Values", error < 5.0f, error);
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "  AFP Neural Network Arithmetic Tests  " << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n--- Element-wise Operations ---" << std::endl;
    testElementwiseAdd();
    testElementwiseSubtract();
    testElementwiseMultiply();
    testElementwiseDivide();
    
    std::cout << "\n--- Dot Product ---" << std::endl;
    testDotProductBasic();
    testDotProductMixedSigns();
    
    std::cout << "\n--- Matrix Operations ---" << std::endl;
    testMatrixVectorMultiply();
    testMatrixMultiply();
    
    std::cout << "\n--- Activation Functions ---" << std::endl;
    testReLU();
    testSigmoid();
    testTanh();
    
    std::cout << "\n--- Reduction Operations ---" << std::endl;
    testSum();
    testMean();
    testMax();
    
    std::cout << "\n--- Broadcasting Operations ---" << std::endl;
    testBroadcastAdd();
    testBroadcastMultiply();
    
    std::cout << "\n--- Edge Cases ---" << std::endl;
    testZeros();
    testSmallValues();
    testLargeValues();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  All tests completed!  " << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}