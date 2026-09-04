#include "../include/afp_encoded_tensor.hpp"
#include "../include/afp_math.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>


static int tests_run = 0;
static int tests_passed = 0;


void expectTrue(
    bool condition,
    const std::string &message
)
{
    ++tests_run;

    if (!condition)
    {
        std::cerr
            << "[FAIL] "
            << message
            << '\n';

        return;
    }

    ++tests_passed;

    std::cout
        << "[PASS] "
        << message
        << '\n';
}


void expectNear(
    float actual,
    float expected,
    float tolerance,
    const std::string &message
)
{
    ++tests_run;

    const float difference =
        std::fabs(actual - expected);

    if (difference > tolerance)
    {
        std::cerr
            << "[FAIL] "
            << message
            << "\n"
            << "  Expected: "
            << expected
            << "\n"
            << "  Actual:   "
            << actual
            << "\n"
            << "  Difference: "
            << difference
            << "\n";

        return;
    }

    ++tests_passed;

    std::cout
        << "[PASS] "
        << message
        << '\n';
}


void expectTensorSize(
    const AFPEncodedTensor &tensor,
    std::size_t expected_size,
    const std::string &message
)
{
    expectTrue(
        tensor.size() == expected_size,
        message
    );
}


void expectTensorNear(
    const std::vector<float> &actual,
    const std::vector<float> &expected,
    float tolerance,
    const std::string &message
)
{
    expectTrue(
        actual.size() == expected.size(),
        message + " size"
    );

    if (actual.size() != expected.size())
    {
        return;
    }

    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        expectNear(
            actual[i],
            expected[i],
            tolerance,
            message + " element " +
            std::to_string(i)
        );
    }
}


/*
 * Helper functions.
 *
 * Adjust these if your AFPQuantizer API differs.
 */

AFPEncodedTensor makeTensor(
    const std::vector<float> &values,
    const AFPConfig &config
)
{
    AFPQuantizer quantizer(config);

    return quantizer.encode(values);
}


std::vector<float> decodeTensor(
    const AFPEncodedTensor &tensor
)
{
    AFPQuantizer quantizer(tensor.config());

    return quantizer.decode(tensor);
}


/*
 * Element-wise arithmetic tests.
 */

void testAdd(
    const AFPConfig &config
)
{
    std::cout << "\n=== ADD ===\n";

    const auto a =
        makeTensor(
            {1.0f, 2.0f, -3.0f, 4.0f},
            config
        );

    const auto b =
        makeTensor(
            {2.0f, -1.0f, 3.0f, 0.5f},
            config
        );

    const auto result =
        AFPArithmetic::add(a, b);

    const auto decoded =
        decodeTensor(result);

    expectTensorNear(
        decoded,
        {3.0f, 1.0f, 0.0f, 4.5f},
        0.5f,
        "add"
    );
}


void testSubtract(
    const AFPConfig &config
)
{
    std::cout << "\n=== SUBTRACT ===\n";

    const auto a =
        makeTensor(
            {5.0f, 2.0f, -3.0f},
            config
        );

    const auto b =
        makeTensor(
            {2.0f, 5.0f, 1.0f},
            config
        );

    const auto result =
        AFPArithmetic::subtract(a, b);

    const auto decoded =
        decodeTensor(result);

    expectTensorNear(
        decoded,
        {3.0f, -3.0f, -4.0f},
        0.5f,
        "subtract"
    );
}


void testMultiply(
    const AFPConfig &config
)
{
    std::cout << "\n=== MULTIPLY ===\n";

    const auto a =
        makeTensor(
            {2.0f, -3.0f, 0.5f},
            config
        );

    const auto b =
        makeTensor(
            {4.0f, 2.0f, 8.0f},
            config
        );

    const auto result =
        AFPArithmetic::multiply(a, b);

    const auto decoded =
        decodeTensor(result);

    expectTensorNear(
        decoded,
        {8.0f, -6.0f, 4.0f},
        0.75f,
        "multiply"
    );
}


void testDivide(
    const AFPConfig &config
)
{
    std::cout << "\n=== DIVIDE ===\n";

    const auto a =
        makeTensor(
            {8.0f, -9.0f, 1.0f, 7.0f},
            config
        );

    const auto b =
        makeTensor(
            {2.0f, 3.0f, 4.0f, 2.0f},
            config
        );

    const auto result =
        AFPArithmetic::divide(a, b);

    const auto decoded =
        decodeTensor(result);

    expectTensorNear(
        decoded,
        {4.0f, -3.0f, 0.25f, 3.5f},
        0.5f,
        "divide"
    );
}


/*
 * Unary operations.
 */

void testNegate(
    const AFPConfig &config
)
{
    std::cout << "\n=== NEGATE ===\n";

    const auto input =
        makeTensor(
            {1.0f, -2.0f, 0.0f, 3.5f},
            config
        );

    const auto result =
        AFPArithmetic::negate(input);

    expectTensorNear(
        decodeTensor(result),
        {-1.0f, 2.0f, 0.0f, -3.5f},
        0.5f,
        "negate"
    );
}


void testAbsolute(
    const AFPConfig &config
)
{
    std::cout << "\n=== ABSOLUTE ===\n";

    const auto input =
        makeTensor(
            {-4.0f, -1.0f, 0.0f, 2.0f},
            config
        );

    const auto result =
        AFPArithmetic::absolute(input);

    expectTensorNear(
        decodeTensor(result),
        {4.0f, 1.0f, 0.0f, 2.0f},
        0.5f,
        "absolute"
    );
}


/*
 * Comparison operations.
 */

void testMinimum(
    const AFPConfig &config
)
{
    std::cout << "\n=== MINIMUM ===\n";

    const auto a =
        makeTensor(
            {1.0f, 5.0f, -2.0f},
            config
        );

    const auto b =
        makeTensor(
            {2.0f, 3.0f, -4.0f},
            config
        );

    const auto result =
        AFPArithmetic::minimum(a, b);

    expectTensorNear(
        decodeTensor(result),
        {1.0f, 3.0f, -4.0f},
        0.5f,
        "minimum"
    );
}


void testMaximum(
    const AFPConfig &config
)
{
    std::cout << "\n=== MAXIMUM ===\n";

    const auto a =
        makeTensor(
            {1.0f, 5.0f, -2.0f},
            config
        );

    const auto b =
        makeTensor(
            {2.0f, 3.0f, -4.0f},
            config
        );

    const auto result =
        AFPArithmetic::maximum(a, b);

    expectTensorNear(
        decodeTensor(result),
        {2.0f, 5.0f, -2.0f},
        0.5f,
        "maximum"
    );
}


void testClamp(
    const AFPConfig &config
)
{
    std::cout << "\n=== CLAMP ===\n";

    const auto input =
        makeTensor(
            {-5.0f, -1.0f, 0.0f, 2.0f, 10.0f},
            config
        );

    const auto result =
        AFPArithmetic::clamp(
            input,
            makeTensor(
                {-1.0f},
                config
            ),
            makeTensor(
                {3.0f},
                config
            )
        );

    expectTensorNear(
        decodeTensor(result),
        {-1.0f, -1.0f, 0.0f, 2.0f, 3.0f},
        0.5f,
        "clamp"
    );
}


/*
 * Activation tests.
 */

void testReLU(
    const AFPConfig &config
)
{
    std::cout << "\n=== RELU ===\n";

    const auto input =
        makeTensor(
            {-2.0f, -0.5f, 0.0f, 1.0f, 4.0f},
            config
        );

    const auto result =
        AFPArithmetic::relu(input);

    expectTensorNear(
        decodeTensor(result),
        {0.0f, 0.0f, 0.0f, 1.0f, 4.0f},
        0.5f,
        "relu"
    );
}


void testLeakyReLU(
    const AFPConfig &config
)
{
    std::cout << "\n=== LEAKY RELU ===\n";

    const auto input =
        makeTensor(
            {-2.0f, -1.0f, 0.0f, 2.0f},
            config
        );

    const auto result =
        AFPArithmetic::leakyRelu(
            input,
            makeTensor(
                {0.1f},
                config
            )
        );

    expectTensorNear(
        decodeTensor(result),
        {-0.2f, -0.1f, 0.0f, 2.0f},
        0.35f,
        "leakyRelu"
    );
}


void testTanh(
    const AFPConfig &config
)
{
    std::cout << "\n=== TANH ===\n";

    const auto input =
        makeTensor(
            {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f},
            config
        );

    const auto result =
        AFPArithmetic::tanh(input);

    expectTensorNear(
        decodeTensor(result),
        {
            std::tanh(-2.0f),
            std::tanh(-1.0f),
            0.0f,
            std::tanh(1.0f),
            std::tanh(2.0f)
        },
        0.25f,
        "tanh"
    );
}


void testSigmoid(
    const AFPConfig &config
)
{
    std::cout << "\n=== SIGMOID ===\n";

    const auto input =
        makeTensor(
            {-2.0f, 0.0f, 2.0f},
            config
        );

    const auto result =
        AFPArithmetic::sigmoid(input);

    expectTensorNear(
        decodeTensor(result),
        {
            1.0f / (1.0f + std::exp(2.0f)),
            0.5f,
            1.0f / (1.0f + std::exp(-2.0f))
        },
        0.25f,
        "sigmoid"
    );
}


/*
 * Reduction tests.
 */

void testSum(
    const AFPConfig &config
)
{
    std::cout << "\n=== SUM ===\n";

    const auto input =
        makeTensor(
            {1.0f, 2.0f, 3.0f, 4.0f},
            config
        );

    const auto result =
        AFPArithmetic::sum(input);

    expectTensorSize(
        result,
        1,
        "sum result size"
    );

    const auto decoded =
        decodeTensor(result);

    expectNear(
        decoded[0],
        10.0f,
        0.75f,
        "sum value"
    );
}


void testMean(
    const AFPConfig &config
)
{
    std::cout << "\n=== MEAN ===\n";

    const auto input =
        makeTensor(
            {2.0f, 4.0f, 6.0f, 8.0f},
            config
        );

    const auto result =
        AFPArithmetic::mean(input);

    expectTensorSize(
        result,
        1,
        "mean result size"
    );

    const auto decoded =
        decodeTensor(result);

    expectNear(
        decoded[0],
        5.0f,
        0.75f,
        "mean value"
    );
}


void testMax(
    const AFPConfig &config
)
{
    std::cout << "\n=== MAX ===\n";

    const auto input =
        makeTensor(
            {-3.0f, 2.0f, 7.0f, 1.0f},
            config
        );

    const auto result =
        AFPArithmetic::max(input);

    const auto decoded =
        decodeTensor(result);

    expectNear(
        decoded[0],
        7.0f,
        0.5f,
        "max value"
    );
}


/*
 * Square/reduction tests.
 */

void testSumSquares(
    const AFPConfig &config
)
{
    std::cout << "\n=== SUM SQUARES ===\n";

    const auto input =
        makeTensor(
            {1.0f, 2.0f, 3.0f},
            config
        );

    const auto result =
        AFPArithmetic::sumSquares(input);

    const auto decoded =
        decodeTensor(result);

    expectNear(
        decoded[0],
        14.0f,
        1.0f,
        "sumSquares value"
    );
}


void testRMS(
    const AFPConfig &config
)
{
    std::cout << "\n=== RMS ===\n";

    const auto input =
        makeTensor(
            {3.0f, 4.0f},
            config
        );

    const auto result =
        AFPArithmetic::rms(input);

    const auto decoded =
        decodeTensor(result);

    const float expected =
        std::sqrt(
            (9.0f + 16.0f) / 2.0f
        );

    expectNear(
        decoded[0],
        expected,
        0.75f,
        "rms value"
    );
}


/*
 * Reciprocal and square root.
 */

void testReciprocal(
    const AFPConfig &config
)
{
    std::cout << "\n=== RECIPROCAL ===\n";

    const auto input =
        makeTensor(
            {1.0f, 2.0f, 4.0f, -2.0f},
            config
        );

    const auto result =
        AFPArithmetic::reciprocal(input);

    expectTensorNear(
        decodeTensor(result),
        {1.0f, 0.5f, 0.25f, -0.5f},
        0.3f,
        "reciprocal"
    );
}


void testSqrt(
    const AFPConfig &config
)
{
    std::cout << "\n=== SQRT ===\n";

    const auto input =
        makeTensor(
            {0.0f, 1.0f, 4.0f, 9.0f},
            config
        );

    const auto result =
        AFPArithmetic::sqrt(input);

    expectTensorNear(
        decodeTensor(result),
        {0.0f, 1.0f, 2.0f, 3.0f},
        0.5f,
        "sqrt"
    );
}


/*
 * Matrix operations.
 */

void testDotProduct(
    const AFPConfig &config
)
{
    std::cout << "\n=== DOT PRODUCT ===\n";

    const auto a =
        makeTensor(
            {1.0f, 2.0f, 3.0f},
            config
        );

    const auto b =
        makeTensor(
            {4.0f, 5.0f, 6.0f},
            config
        );

    const auto result =
        AFPArithmetic::dotProduct(a, b);

    const auto decoded =
        decodeTensor(result);

    expectTensorSize(
        result,
        1,
        "dot product result size"
    );

    expectNear(
        decoded[0],
        32.0f,
        2.0f,
        "dot product value"
    );
}


void testMatrixVectorMultiply(
    const AFPConfig &config
)
{
    std::cout
        << "\n=== MATRIX VECTOR MULTIPLY ===\n";

    /*
     * Matrix:
     *
     * [ 1  2 ]
     * [ 3  4 ]
     */

    const auto weights =
        makeTensor(
            {
                1.0f, 2.0f,
                3.0f, 4.0f
            },
            config
        );

    /*
     * Vector:
     *
     * [ 5 ]
     * [ 6 ]
     */

    const auto input =
        makeTensor(
            {5.0f, 6.0f},
            config
        );

    const auto result =
        AFPArithmetic::matrixVectorMultiply(
            weights,
            input,
            2,
            2
        );

    expectTensorNear(
        decodeTensor(result),
        {
            17.0f,
            39.0f
        },
        2.0f,
        "matrixVectorMultiply"
    );
}


void testMatrixMultiply(
    const AFPConfig &config
)
{
    std::cout
        << "\n=== MATRIX MULTIPLY ===\n";

    /*
     * A =
     *
     * [ 1  2 ]
     * [ 3  4 ]
     *
     * B =
     *
     * [ 5  6 ]
     * [ 7  8 ]
     */

    const auto a =
        makeTensor(
            {
                1.0f, 2.0f,
                3.0f, 4.0f
            },
            config
        );

    const auto b =
        makeTensor(
            {
                5.0f, 6.0f,
                7.0f, 8.0f
            },
            config
        );

    const auto result =
        AFPArithmetic::matrixMultiply(
            a,
            b,
            2,
            2,
            2
        );

    expectTensorNear(
        decodeTensor(result),
        {
            19.0f, 22.0f,
            43.0f, 50.0f
        },
        3.0f,
        "matrixMultiply"
    );
}


/*
 * Outer product.
 */

void testOuterProduct(
    const AFPConfig &config
)
{
    std::cout
        << "\n=== OUTER PRODUCT ===\n";

    const auto a =
        makeTensor(
            {1.0f, 2.0f},
            config
        );

    const auto b =
        makeTensor(
            {3.0f, 4.0f, 5.0f},
            config
        );

    const auto result =
        AFPArithmetic::outerProduct(
            a,
            b
        );

    expectTensorNear(
        decodeTensor(result),
        {
            3.0f, 4.0f, 5.0f,
            6.0f, 8.0f, 10.0f
        },
        1.0f,
        "outerProduct"
    );
}


/*
 * Transpose.
 */

void testTranspose(
    const AFPConfig &config
)
{
    std::cout
        << "\n=== TRANSPOSE ===\n";

    /*
     * Original:
     *
     * [ 1 2 3 ]
     * [ 4 5 6 ]
     *
     * Transposed:
     *
     * [ 1 4 ]
     * [ 2 5 ]
     * [ 3 6 ]
     */

    const auto input =
        makeTensor(
            {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
            },
            config
        );

    const auto result =
        AFPArithmetic::transpose(
            input,
            2,
            3
        );

    expectTensorNear(
        decodeTensor(result),
        {
            1.0f, 4.0f,
            2.0f, 5.0f,
            3.0f, 6.0f
        },
        0.5f,
        "transpose"
    );
}


/*
 * Exponential.
 */

void testExp(
    const AFPConfig &config
)
{
    std::cout << "\n=== EXP ===\n";

    const auto input =
        makeTensor(
            {-1.0f, 0.0f, 1.0f},
            config
        );

    const auto result =
        AFPArithmetic::exp(input);

    expectTensorNear(
        decodeTensor(result),
        {
            std::exp(-1.0f),
            1.0f,
            std::exp(1.0f)
        },
        0.5f,
        "exp"
    );
}


/*
 * Softmax.
 */

void testSoftmax(
    const AFPConfig &config
)
{
    std::cout << "\n=== SOFTMAX ===\n";

    const auto input =
        makeTensor(
            {1.0f, 2.0f, 3.0f},
            config
        );

    const auto result =
        AFPArithmetic::softmax(input);

    const auto decoded =
        decodeTensor(result);

    expectTensorSize(
        result,
        3,
        "softmax result size"
    );

    float sum = 0.0f;

    for (float value : decoded)
    {
        sum += value;
    }

    expectNear(
        sum,
        1.0f,
        0.25f,
        "softmax sums approximately to one"
    );

    expectTrue(
        decoded[2] > decoded[1],
        "softmax ordering 2 > 1"
    );

    expectTrue(
        decoded[1] > decoded[0],
        "softmax ordering 1 > 0"
    );
}


/*
 * Main.
 */

int main()
{
    AFPConfig config;

    /*
     * Configure this to match your AFP implementation.
     *
     * Adjust these fields if AFPConfig uses different
     * names.
     */

    std::cout
        << "================================\n"
        << "AFP MATH TESTS\n"
        << "================================\n";

    testAdd(config);
    testSubtract(config);
    testMultiply(config);
    testDivide(config);

    testNegate(config);
    testAbsolute(config);

    testMinimum(config);
    testMaximum(config);
    testClamp(config);

    testReLU(config);
    testLeakyReLU(config);
    testSigmoid(config);
    testTanh(config);

    testSum(config);
    testMean(config);
    testMax(config);

    testSumSquares(config);
    testReciprocal(config);
    testSqrt(config);
    testRMS(config);

    testDotProduct(config);
    testMatrixVectorMultiply(config);
    testMatrixMultiply(config);

    testOuterProduct(config);
    testTranspose(config);

    testExp(config);
    testSoftmax(config);

    std::cout
        << "\n================================\n"
        << "RESULTS\n"
        << "================================\n"
        << "Passed: "
        << tests_passed
        << " / "
        << tests_run
        << '\n';

    return
        tests_passed == tests_run
            ? EXIT_SUCCESS
            : EXIT_FAILURE;
}