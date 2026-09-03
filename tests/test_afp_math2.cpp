#include "../include/afp.hpp"
#include "../include/afp_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>


// ============================================================================
// Configuration
// ============================================================================

namespace
{

constexpr float absolute_tolerance =
    0.0001f;

constexpr float relative_tolerance =
    0.05f;

constexpr std::size_t random_test_count =
    100;

constexpr std::size_t block_size =
    16;


// ============================================================================
// Test Statistics
// ============================================================================

struct TestStatistics
{
    std::size_t passed = 0;
    std::size_t failed = 0;
};

TestStatistics statistics;


// ============================================================================
// Utility Functions
// ============================================================================

bool approximatelyEqual(
    float actual,
    float expected,
    float abs_tolerance = absolute_tolerance,
    float rel_tolerance = relative_tolerance)
{
    const float absolute_error =
        std::fabs(actual - expected);

    if (absolute_error <= abs_tolerance)
    {
        return true;
    }

    const float scale =
        std::max(
            std::fabs(actual),
            std::fabs(expected));

    if (scale == 0.0f)
    {
        return absolute_error <= abs_tolerance;
    }

    return
        absolute_error <=
        rel_tolerance * scale;
}


float maximumAbsoluteError(
    const std::vector<float> &actual,
    const std::vector<float> &expected)
{
    const std::size_t count =
        std::min(
            actual.size(),
            expected.size());

    float maximum_error = 0.0f;

    for (std::size_t i = 0;
         i < count;
         ++i)
    {
        const float error =
            std::fabs(
                actual[i] -
                expected[i]);

        if (error > maximum_error)
        {
            maximum_error = error;
        }
    }

    return maximum_error;
}


float maximumRelativeError(
    const std::vector<float> &actual,
    const std::vector<float> &expected)
{
    const std::size_t count =
        std::min(
            actual.size(),
            expected.size());

    float maximum_error = 0.0f;

    for (std::size_t i = 0;
         i < count;
         ++i)
    {
        const float denominator =
            std::max(
                std::fabs(expected[i]),
                0.000001f);

        const float error =
            std::fabs(
                actual[i] -
                expected[i]) /
            denominator;

        if (error > maximum_error)
        {
            maximum_error = error;
        }
    }

    return maximum_error;
}


bool vectorsApproximatelyEqual(
    const std::vector<float> &actual,
    const std::vector<float> &expected,
    float abs_tolerance = absolute_tolerance,
    float rel_tolerance = relative_tolerance)
{
    if (actual.size() != expected.size())
    {
        return false;
    }

    for (std::size_t i = 0;
         i < actual.size();
         ++i)
    {
        if (!approximatelyEqual(
                actual[i],
                expected[i],
                abs_tolerance,
                rel_tolerance))
        {
            return false;
        }
    }

    return true;
}


void reportTest(
    const std::string &name,
    bool passed)
{
    std::cout
        << std::left
        << std::setw(45)
        << name;

    if (passed)
    {
        std::cout << "PASS\n";

        ++statistics.passed;
    }
    else
    {
        std::cout << "FAIL\n";

        ++statistics.failed;
    }
}


void printVector(
    const std::string &name,
    const std::vector<float> &values)
{
    std::cout << name << ": [";

    for (std::size_t i = 0;
         i < values.size();
         ++i)
    {
        std::cout
            << std::fixed
            << std::setprecision(6)
            << values[i];

        if (i + 1 < values.size())
        {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";
}


// ============================================================================
// FP32 Reference Operations
// ============================================================================

std::vector<float> referenceAdd(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    std::vector<float> result(
        a.size());

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        result[i] =
            a[i] +
            b[i];
    }

    return result;
}


std::vector<float> referenceSubtract(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    std::vector<float> result(
        a.size());

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        result[i] =
            a[i] -
            b[i];
    }

    return result;
}


std::vector<float> referenceMultiply(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    std::vector<float> result(
        a.size());

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        result[i] =
            a[i] *
            b[i];
    }

    return result;
}


std::vector<float> referenceDivide(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    std::vector<float> result(
        a.size());

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        result[i] =
            a[i] /
            b[i];
    }

    return result;
}


float referenceDotProduct(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    float result = 0.0f;

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        result +=
            a[i] *
            b[i];
    }

    return result;
}


std::vector<float> referenceRelu(
    const std::vector<float> &input)
{
    std::vector<float> result(
        input.size());

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        result[i] =
            std::max(
                0.0f,
                input[i]);
    }

    return result;
}


std::vector<float> referenceMatrixVectorMultiply(
    const std::vector<float> &matrix,
    const std::vector<float> &vector,
    std::size_t rows,
    std::size_t columns)
{
    std::vector<float> result(
        rows,
        0.0f);

    for (std::size_t row = 0;
         row < rows;
         ++row)
    {
        for (std::size_t column = 0;
             column < columns;
             ++column)
        {
            result[row] +=
                matrix[
                    row * columns +
                    column] *
                vector[column];
        }
    }

    return result;
}


std::vector<float> referenceMatrixMultiply(
    const std::vector<float> &a,
    const std::vector<float> &b,
    std::size_t rows_a,
    std::size_t columns_a,
    std::size_t columns_b)
{
    std::vector<float> result(
        rows_a * columns_b,
        0.0f);

    for (std::size_t row = 0;
         row < rows_a;
         ++row)
    {
        for (std::size_t column = 0;
             column < columns_b;
             ++column)
        {
            float sum = 0.0f;

            for (std::size_t k = 0;
                 k < columns_a;
                 ++k)
            {
                sum +=
                    a[
                        row * columns_a +
                        k] *
                    b[
                        k * columns_b +
                        column];
            }

            result[
                row * columns_b +
                column] =
                sum;
        }
    }

    return result;
}


// ============================================================================
// AFP Reference Baseline
//
// IMPORTANT:
//
// Compare AFP arithmetic against operations performed on the
// DECODED AFP INPUTS.
//
// This isolates arithmetic error from initial quantization error.
// ============================================================================

struct QuantizedInputs
{
    AFPEncodedTensor encoded_a;
    AFPEncodedTensor encoded_b;

    std::vector<float> decoded_a;
    std::vector<float> decoded_b;
};


QuantizedInputs quantizeInputs(
    AFPQuantizer &quantizer,
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    QuantizedInputs result
    {
        quantizer.encode(a),
        quantizer.encode(b),
        {},
        {}
    };

    result.decoded_a =
        quantizer.decode(
            result.encoded_a);

    result.decoded_b =
        quantizer.decode(
            result.encoded_b);

    return result;
}


// ============================================================================
// Element-wise Tests
// ============================================================================

void testElementwiseAddBasic()
{
    AFPQuantizer quantizer;

    std::vector<float> a
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    std::vector<float> b
    {
        0.5f, 1.5f, 2.5f, 3.5f,
        4.5f, 5.5f, 6.5f, 7.5f,
        8.5f, 9.5f, 10.5f, 11.5f,
        12.5f, 13.5f, 14.5f, 15.5f
    };

    QuantizedInputs inputs =
        quantizeInputs(
            quantizer,
            a,
            b);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::add(
            inputs.encoded_a,
            inputs.encoded_b);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    std::vector<float> expected =
        referenceAdd(
            inputs.decoded_a,
            inputs.decoded_b);

    const bool passed =
        vectorsApproximatelyEqual(
            actual,
            expected,
            0.5f,
            0.05f);

    reportTest(
        "Element-wise add: basic",
        passed);
}


void testElementwiseAddCancellation()
{
    AFPQuantizer quantizer;

    std::vector<float> a
    {
        100.0f, -100.0f,
        50.0f, -50.0f,
        10.0f, -10.0f,
        1.0f, -1.0f,
        0.5f, -0.5f,
        0.25f, -0.25f,
        5.0f, -5.0f,
        2.0f, -2.0f
    };

    std::vector<float> b
    {
        -99.0f, 99.0f,
        -49.0f, 49.0f,
        -9.0f, 9.0f,
        -0.5f, 0.5f,
        -0.25f, 0.25f,
        -0.125f, 0.125f,
        -4.0f, 4.0f,
        -1.0f, 1.0f
    };

    QuantizedInputs inputs =
        quantizeInputs(
            quantizer,
            a,
            b);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::add(
            inputs.encoded_a,
            inputs.encoded_b);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    std::vector<float> expected =
        referenceAdd(
            inputs.decoded_a,
            inputs.decoded_b);

    const bool passed =
        vectorsApproximatelyEqual(
            actual,
            expected,
            1.0f,
            0.10f);

    reportTest(
        "Element-wise add: cancellation",
        passed);
}


void testElementwiseSubtract()
{
    AFPQuantizer quantizer;

    std::vector<float> a
    {
        10.0f, 9.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 4.0f, 3.0f,
        2.0f, 1.0f, 0.5f, 0.25f,
        -1.0f, -2.0f, -3.0f, -4.0f
    };

    std::vector<float> b
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 0.25f, 0.125f,
        -0.5f, -1.0f, -2.0f, -3.0f
    };

    QuantizedInputs inputs =
        quantizeInputs(
            quantizer,
            a,
            b);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::subtract(
            inputs.encoded_a,
            inputs.encoded_b);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    std::vector<float> expected =
        referenceSubtract(
            inputs.decoded_a,
            inputs.decoded_b);

    const bool passed =
        vectorsApproximatelyEqual(
            actual,
            expected,
            0.5f,
            0.05f);

    reportTest(
        "Element-wise subtract",
        passed);
}


void testElementwiseMultiply()
{
    AFPQuantizer quantizer;

    std::vector<float> a
    {
        1.0f, -2.0f, 3.0f, -4.0f,
        5.0f, -6.0f, 7.0f, -8.0f,
        0.5f, -0.5f, 0.25f, -0.25f,
        10.0f, -10.0f, 2.0f, -2.0f
    };

    std::vector<float> b
    {
        -2.0f, -2.0f, 2.0f, 2.0f,
        0.5f, 0.5f, -0.5f, -0.5f,
        4.0f, 4.0f, -4.0f, -4.0f,
        0.25f, 0.25f, -2.0f, -2.0f
    };

    QuantizedInputs inputs =
        quantizeInputs(
            quantizer,
            a,
            b);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::multiply(
            inputs.encoded_a,
            inputs.encoded_b);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    std::vector<float> expected =
        referenceMultiply(
            inputs.decoded_a,
            inputs.decoded_b);

    const bool passed =
        vectorsApproximatelyEqual(
            actual,
            expected,
            1.0f,
            0.10f);

    reportTest(
        "Element-wise multiply",
        passed);
}


void testElementwiseDivide()
{
    AFPQuantizer quantizer;

    std::vector<float> a
    {
        8.0f, -8.0f, 6.0f, -6.0f,
        4.0f, -4.0f, 2.0f, -2.0f,
        16.0f, -16.0f, 32.0f, -32.0f,
        1.0f, -1.0f, 0.5f, -0.5f
    };

    std::vector<float> b
    {
        2.0f, 2.0f, -2.0f, -2.0f,
        4.0f, 4.0f, -4.0f, -4.0f,
        8.0f, 8.0f, 16.0f, 16.0f,
        0.5f, 0.5f, 0.25f, 0.25f
    };

    QuantizedInputs inputs =
        quantizeInputs(
            quantizer,
            a,
            b);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::divide(
            inputs.encoded_a,
            inputs.encoded_b);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    std::vector<float> expected =
        referenceDivide(
            inputs.decoded_a,
            inputs.decoded_b);

    const bool passed =
        vectorsApproximatelyEqual(
            actual,
            expected,
            1.0f,
            0.15f);

    reportTest(
        "Element-wise divide",
        passed);
}


// ============================================================================
// Dot Product Tests
// ============================================================================

void testDotProductBasic()
{
    AFPQuantizer quantizer;

    std::vector<float> a
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    std::vector<float> b(
        block_size,
        2.0f);

    QuantizedInputs inputs =
        quantizeInputs(
            quantizer,
            a,
            b);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::dotProduct(
            inputs.encoded_a,
            inputs.encoded_b);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    const float expected =
        referenceDotProduct(
            inputs.decoded_a,
            inputs.decoded_b);

    bool passed =
        actual.size() == 1;

    if (passed)
    {
        passed =
            approximatelyEqual(
                actual[0],
                expected,
                5.0f,
                0.05f);
    }

    reportTest(
        "Dot product: basic",
        passed);
}


void testDotProductMixedSigns()
{
    AFPQuantizer quantizer;

    std::vector<float> a
    {
        1.0f, -2.0f, 3.0f, -4.0f,
        5.0f, -6.0f, 7.0f, -8.0f,
        9.0f, -10.0f, 11.0f, -12.0f,
        13.0f, -14.0f, 15.0f, -16.0f
    };

    std::vector<float> b
    {
        -1.0f, 1.0f, -1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f, 1.0f
    };

    QuantizedInputs inputs =
        quantizeInputs(
            quantizer,
            a,
            b);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::dotProduct(
            inputs.encoded_a,
            inputs.encoded_b);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    const float expected =
        referenceDotProduct(
            inputs.decoded_a,
            inputs.decoded_b);

    bool passed =
        actual.size() == 1;

    if (passed)
    {
        passed =
            approximatelyEqual(
                actual[0],
                expected,
                5.0f,
                0.05f);
    }

    reportTest(
        "Dot product: mixed signs",
        passed);
}


void testDotProductCancellation()
{
    AFPQuantizer quantizer;

    std::vector<float> a
    {
        100.0f, -100.0f,
        50.0f, -50.0f,
        25.0f, -25.0f,
        10.0f, -10.0f,
        5.0f, -5.0f,
        2.0f, -2.0f,
        1.0f, -1.0f,
        0.5f, -0.5f
    };

    std::vector<float> b(
        block_size,
        1.0f);

    QuantizedInputs inputs =
        quantizeInputs(
            quantizer,
            a,
            b);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::dotProduct(
            inputs.encoded_a,
            inputs.encoded_b);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    const float expected =
        referenceDotProduct(
            inputs.decoded_a,
            inputs.decoded_b);

    bool passed =
        actual.size() == 1;

    if (passed)
    {
        passed =
            approximatelyEqual(
                actual[0],
                expected,
                2.0f,
                0.20f);
    }

    reportTest(
        "Dot product: cancellation",
        passed);
}


void testDotProductTwoBlocks()
{
    AFPQuantizer quantizer;

    std::vector<float> a(
        32);

    std::vector<float> b(
        32);

    for (std::size_t i = 0;
         i < 32;
         ++i)
    {
        a[i] =
            static_cast<float>(
                i + 1);

        b[i] =
            (i % 2 == 0)
                ? 1.0f
                : -1.0f;
    }

    QuantizedInputs inputs =
        quantizeInputs(
            quantizer,
            a,
            b);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::dotProduct(
            inputs.encoded_a,
            inputs.encoded_b);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    const float expected =
        referenceDotProduct(
            inputs.decoded_a,
            inputs.decoded_b);

    bool passed =
        actual.size() == 1;

    if (passed)
    {
        passed =
            approximatelyEqual(
                actual[0],
                expected,
                10.0f,
                0.10f);
    }

    reportTest(
        "Dot product: two blocks",
        passed);
}


// ============================================================================
// ReLU Tests
// ============================================================================

void testRelu()
{
    AFPQuantizer quantizer;

    std::vector<float> input
    {
        -10.0f, -2.0f, -1.0f, -0.5f,
        -0.1f, 0.0f, 0.1f, 0.5f,
        1.0f, 2.0f, 10.0f, -5.0f,
        3.0f, -3.0f, 0.25f, -0.25f
    };

    AFPEncodedTensor encoded_input =
        quantizer.encode(
            input);

    std::vector<float> decoded_input =
        quantizer.decode(
            encoded_input);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::relu(
            encoded_input);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    std::vector<float> expected =
        referenceRelu(
            decoded_input);

    const bool passed =
        vectorsApproximatelyEqual(
            actual,
            expected,
            0.25f,
            0.05f);

    reportTest(
        "ReLU",
        passed);
}


// ============================================================================
// Sigmoid / Tanh Tests
// ============================================================================

void testSigmoidRange()
{
    AFPQuantizer quantizer;

    std::vector<float> input
    {
        -4.0f, -3.0f, -2.0f, -1.0f,
        -0.5f, 0.0f, 0.5f, 1.0f,
        2.0f, 3.0f, 4.0f, -0.25f,
        0.25f, 1.5f, -1.5f, 0.75f
    };

    AFPEncodedTensor encoded_input =
        quantizer.encode(
            input);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::sigmoid(
            encoded_input);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    bool passed =
        actual.size() ==
        input.size();

    for (float value : actual)
    {
        if (value < -0.1f ||
            value > 1.1f)
        {
            passed = false;
            break;
        }
    }

    reportTest(
        "Sigmoid: output range",
        passed);
}


void testTanhRange()
{
    AFPQuantizer quantizer;

    std::vector<float> input
    {
        -4.0f, -3.0f, -2.0f, -1.0f,
        -0.5f, 0.0f, 0.5f, 1.0f,
        2.0f, 3.0f, 4.0f, -0.25f,
        0.25f, 1.5f, -1.5f, 0.75f
    };

    AFPEncodedTensor encoded_input =
        quantizer.encode(
            input);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::tanh(
            encoded_input);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    bool passed =
        actual.size() ==
        input.size();

    for (float value : actual)
    {
        if (value < -1.1f ||
            value > 1.1f)
        {
            passed = false;
            break;
        }
    }

    reportTest(
        "Tanh: output range",
        passed);
}


// ============================================================================
// Matrix Tests
// ============================================================================

void testMatrixVectorMultiply()
{
    AFPQuantizer quantizer;

    constexpr std::size_t rows =
        4;

    constexpr std::size_t columns =
        4;

    std::vector<float> matrix
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    std::vector<float> vector
    {
        1.0f,
        -1.0f,
        2.0f,
        0.5f
    };

    AFPEncodedTensor encoded_matrix =
        quantizer.encode(
            matrix);

    AFPEncodedTensor encoded_vector =
        quantizer.encode(
            vector);

    std::vector<float> decoded_matrix =
        quantizer.decode(
            encoded_matrix);

    std::vector<float> decoded_vector =
        quantizer.decode(
            encoded_vector);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::matrixVectorMultiply(
            encoded_matrix,
            encoded_vector,
            rows,
            columns);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    std::vector<float> expected =
        referenceMatrixVectorMultiply(
            decoded_matrix,
            decoded_vector,
            rows,
            columns);

    const bool passed =
        vectorsApproximatelyEqual(
            actual,
            expected,
            5.0f,
            0.10f);

    reportTest(
        "Matrix-vector multiply",
        passed);
}


void testMatrixMultiply()
{
    AFPQuantizer quantizer;

    constexpr std::size_t rows_a =
        2;

    constexpr std::size_t columns_a =
        2;

    constexpr std::size_t columns_b =
        2;

    std::vector<float> a
    {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    std::vector<float> b
    {
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    AFPEncodedTensor encoded_a =
        quantizer.encode(a);

    AFPEncodedTensor encoded_b =
        quantizer.encode(b);

    std::vector<float> decoded_a =
        quantizer.decode(
            encoded_a);

    std::vector<float> decoded_b =
        quantizer.decode(
            encoded_b);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::matrixMultiply(
            encoded_a,
            encoded_b,
            rows_a,
            columns_a,
            columns_b);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    std::vector<float> expected =
        referenceMatrixMultiply(
            decoded_a,
            decoded_b,
            rows_a,
            columns_a,
            columns_b);

    const bool passed =
        vectorsApproximatelyEqual(
            actual,
            expected,
            10.0f,
            0.15f);

    reportTest(
        "Matrix multiply",
        passed);
}


// ============================================================================
// Reduction Tests
// ============================================================================

void testSum()
{
    AFPQuantizer quantizer;

    std::vector<float> input
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    AFPEncodedTensor encoded_input =
        quantizer.encode(
            input);

    std::vector<float> decoded_input =
        quantizer.decode(
            encoded_input);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::sum(
            encoded_input);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    float expected = 0.0f;

    for (float value : decoded_input)
    {
        expected += value;
    }

    bool passed =
        actual.size() == 1;

    if (passed)
    {
        passed =
            approximatelyEqual(
                actual[0],
                expected,
                10.0f,
                0.05f);
    }

    reportTest(
        "Reduction: sum",
        passed);
}


void testMean()
{
    AFPQuantizer quantizer;

    std::vector<float> input
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    AFPEncodedTensor encoded_input =
        quantizer.encode(
            input);

    std::vector<float> decoded_input =
        quantizer.decode(
            encoded_input);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::mean(
            encoded_input);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    float expected = 0.0f;

    for (float value : decoded_input)
    {
        expected += value;
    }

    expected /=
        static_cast<float>(
            decoded_input.size());

    bool passed =
        actual.size() == 1;

    if (passed)
    {
        passed =
            approximatelyEqual(
                actual[0],
                expected,
                2.0f,
                0.10f);
    }

    reportTest(
        "Reduction: mean",
        passed);
}


void testMax()
{
    AFPQuantizer quantizer;

    std::vector<float> input
    {
        -10.0f, -5.0f, -1.0f, 0.0f,
        1.0f, 2.0f, 3.0f, 8.0f,
        4.0f, 16.0f, 5.0f, 7.0f,
        6.0f, 10.0f, 12.0f, 9.0f
    };

    AFPEncodedTensor encoded_input =
        quantizer.encode(
            input);

    std::vector<float> decoded_input =
        quantizer.decode(
            encoded_input);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::max(
            encoded_input);

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    float expected =
        decoded_input[0];

    for (float value : decoded_input)
    {
        expected =
            std::max(
                expected,
                value);
    }

    bool passed =
        actual.size() == 1;

    if (passed)
    {
        passed =
            approximatelyEqual(
                actual[0],
                expected,
                1.0f,
                0.05f);
    }

    reportTest(
        "Reduction: max",
        passed);
}


// ============================================================================
// Broadcasting Tests
// ============================================================================

void testBroadcastAdd()
{
    AFPQuantizer quantizer;

    std::vector<float> input
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    std::vector<float> scalar
    {
        2.5f
    };

    AFPEncodedTensor encoded_input =
        quantizer.encode(
            input);

    AFPEncodedTensor encoded_scalar =
        quantizer.encode(
            scalar);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::broadcastAdd(
            encoded_input,
            encoded_scalar,
            input.size());

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    std::vector<float> decoded_input =
        quantizer.decode(
            encoded_input);

    std::vector<float> decoded_scalar =
        quantizer.decode(
            encoded_scalar);

    std::vector<float> expected(
        decoded_input.size());

    for (std::size_t i = 0;
         i < expected.size();
         ++i)
    {
        expected[i] =
            decoded_input[i] +
            decoded_scalar[0];
    }

    const bool passed =
        vectorsApproximatelyEqual(
            actual,
            expected,
            0.5f,
            0.10f);

    reportTest(
        "Broadcast add",
        passed);
}


void testBroadcastMultiply()
{
    AFPQuantizer quantizer;

    std::vector<float> input
    {
        1.0f, -2.0f, 3.0f, -4.0f,
        5.0f, -6.0f, 7.0f, -8.0f,
        9.0f, -10.0f, 11.0f, -12.0f,
        13.0f, -14.0f, 15.0f, -16.0f
    };

    std::vector<float> scalar
    {
        -1.5f
    };

    AFPEncodedTensor encoded_input =
        quantizer.encode(
            input);

    AFPEncodedTensor encoded_scalar =
        quantizer.encode(
            scalar);

    AFPEncodedTensor encoded_result =
        AFPArithmetic::broadcastMultiply(
            encoded_input,
            encoded_scalar,
            input.size());

    std::vector<float> actual =
        quantizer.decode(
            encoded_result);

    std::vector<float> decoded_input =
        quantizer.decode(
            encoded_input);

    std::vector<float> decoded_scalar =
        quantizer.decode(
            encoded_scalar);

    std::vector<float> expected(
        decoded_input.size());

    for (std::size_t i = 0;
         i < expected.size();
         ++i)
    {
        expected[i] =
            decoded_input[i] *
            decoded_scalar[0];
    }

    const bool passed =
        vectorsApproximatelyEqual(
            actual,
            expected,
            1.0f,
            0.10f);

    reportTest(
        "Broadcast multiply",
        passed);
}


// ============================================================================
// Randomized Tests
// ============================================================================

void testRandomElementwiseMultiply()
{
    AFPQuantizer quantizer;

    std::mt19937 generator(
        12345);

    std::uniform_real_distribution<float>
        distribution(
            -10.0f,
            10.0f);

    bool passed = true;

    float worst_error = 0.0f;

    for (std::size_t test = 0;
         test < random_test_count;
         ++test)
    {
        std::vector<float> a(
            block_size);

        std::vector<float> b(
            block_size);

        for (std::size_t i = 0;
             i < block_size;
             ++i)
        {
            a[i] =
                distribution(
                    generator);

            b[i] =
                distribution(
                    generator);
        }

        QuantizedInputs inputs =
            quantizeInputs(
                quantizer,
                a,
                b);

        AFPEncodedTensor encoded_result =
            AFPArithmetic::multiply(
                inputs.encoded_a,
                inputs.encoded_b);

        std::vector<float> actual =
            quantizer.decode(
                encoded_result);

        std::vector<float> expected =
            referenceMultiply(
                inputs.decoded_a,
                inputs.decoded_b);

        const float error =
            maximumAbsoluteError(
                actual,
                expected);

        worst_error =
            std::max(
                worst_error,
                error);

        if (!vectorsApproximatelyEqual(
                actual,
                expected,
                2.0f,
                0.15f))
        {
            passed = false;
            break;
        }
    }

    std::cout
        << std::left
        << std::setw(45)
        << "Random multiply (100 tests)";

    if (passed)
    {
        std::cout
            << "PASS"
            << " (worst error: "
            << worst_error
            << ")\n";

        ++statistics.passed;
    }
    else
    {
        std::cout
            << "FAIL"
            << " (worst error: "
            << worst_error
            << ")\n";

        ++statistics.failed;
    }
}


void testRandomDotProduct()
{
    AFPQuantizer quantizer;

    std::mt19937 generator(
        54321);

    std::uniform_real_distribution<float>
        distribution(
            -5.0f,
            5.0f);

    bool passed = true;

    float worst_error = 0.0f;

    for (std::size_t test = 0;
         test < random_test_count;
         ++test)
    {
        std::vector<float> a(
            block_size);

        std::vector<float> b(
            block_size);

        for (std::size_t i = 0;
             i < block_size;
             ++i)
        {
            a[i] =
                distribution(
                    generator);

            b[i] =
                distribution(
                    generator);
        }

        QuantizedInputs inputs =
            quantizeInputs(
                quantizer,
                a,
                b);

        AFPEncodedTensor encoded_result =
            AFPArithmetic::dotProduct(
                inputs.encoded_a,
                inputs.encoded_b);

        std::vector<float> actual =
            quantizer.decode(
                encoded_result);

        const float expected =
            referenceDotProduct(
                inputs.decoded_a,
                inputs.decoded_b);

        if (actual.size() != 1)
        {
            passed = false;
            break;
        }

        const float error =
            std::fabs(
                actual[0] -
                expected);

        worst_error =
            std::max(
                worst_error,
                error);

        if (!approximatelyEqual(
                actual[0],
                expected,
                5.0f,
                0.15f))
        {
            passed = false;
            break;
        }
    }

    std::cout
        << std::left
        << std::setw(45)
        << "Random dot product (100 tests)";

    if (passed)
    {
        std::cout
            << "PASS"
            << " (worst error: "
            << worst_error
            << ")\n";

        ++statistics.passed;
    }
    else
    {
        std::cout
            << "FAIL"
            << " (worst error: "
            << worst_error
            << ")\n";

        ++statistics.failed;
    }
}


// ============================================================================
// Error Handling Tests
// ============================================================================

void testSizeMismatch()
{
    AFPQuantizer quantizer;

    std::vector<float> a(
        16,
        1.0f);

    std::vector<float> b(
        32,
        1.0f);

    AFPEncodedTensor encoded_a =
        quantizer.encode(a);

    AFPEncodedTensor encoded_b =
        quantizer.encode(b);

    bool threw_exception =
        false;

    try
    {
        AFPArithmetic::add(
            encoded_a,
            encoded_b);
    }
    catch (const std::exception &)
    {
        threw_exception =
            true;
    }

    reportTest(
        "Error handling: size mismatch",
        threw_exception);
}


void testDivisionByZero()
{
    AFPQuantizer quantizer;

    std::vector<float> a(
        block_size,
        1.0f);

    std::vector<float> b(
        block_size,
        1.0f);

    b[5] =
        0.0f;

    AFPEncodedTensor encoded_a =
        quantizer.encode(a);

    AFPEncodedTensor encoded_b =
        quantizer.encode(b);

    bool threw_exception =
        false;

    try
    {
        AFPArithmetic::divide(
            encoded_a,
            encoded_b);
    }
    catch (const std::exception &)
    {
        threw_exception =
            true;
    }

    reportTest(
        "Error handling: division by zero",
        threw_exception);
}


// ============================================================================
// Main
// ============================================================================

} // namespace


int main()
{
    std::cout
        << "============================================================\n";

    std::cout
        << "              AFP Arithmetic Test Suite\n";

    std::cout
        << "============================================================\n";


    std::cout
        << "\n--- Element-wise Operations ---\n";

    testElementwiseAddBasic();
    testElementwiseAddCancellation();
    testElementwiseSubtract();
    testElementwiseMultiply();
    testElementwiseDivide();


    std::cout
        << "\n--- Dot Products ---\n";

    testDotProductBasic();
    testDotProductMixedSigns();
    testDotProductCancellation();
    testDotProductTwoBlocks();


    std::cout
        << "\n--- Activations ---\n";

    testRelu();
    testSigmoidRange();
    testTanhRange();


    std::cout
        << "\n--- Matrix Operations ---\n";

    testMatrixVectorMultiply();
    testMatrixMultiply();


    std::cout
        << "\n--- Reductions ---\n";

    testSum();
    testMean();
    testMax();


    std::cout
        << "\n--- Broadcasting ---\n";

    testBroadcastAdd();
    testBroadcastMultiply();


    std::cout
        << "\n--- Randomized Testing ---\n";

    testRandomElementwiseMultiply();
    testRandomDotProduct();


    std::cout
        << "\n--- Error Handling ---\n";

    testSizeMismatch();
    testDivisionByZero();


    std::cout
        << "\n============================================================\n";

    std::cout
        << "Passed: "
        << statistics.passed
        << "\n";

    std::cout
        << "Failed: "
        << statistics.failed
        << "\n";

    std::cout
        << "Total:  "
        << statistics.passed +
           statistics.failed
        << "\n";

    std::cout
        << "============================================================\n";


    return
        statistics.failed == 0
            ? 0
            : 1;
}