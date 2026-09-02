#include "../include/afp.hpp"
#include "../include/afp_math.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct TestCase
{
    std::string name;
    std::vector<float> a;
    std::vector<float> b;
};

float fp32DotProduct(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    float result = 0.0f;

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result += a[i] * b[i];
    }

    return result;
}

void runTest(const TestCase &test)
{
    AFPQuantizer quantizer;

    AFPEncodedTensor encoded_a = quantizer.encode(test.a);
    AFPEncodedTensor encoded_b = quantizer.encode(test.b);

    const std::vector<float> decoded_a = quantizer.decode(encoded_a);
    const std::vector<float> decoded_b = quantizer.decode(encoded_b);

    const float afp_result =
        AFPArithmetic::dotProduct(encoded_a, encoded_b);

    const float fp32_result =
        fp32DotProduct(decoded_a, decoded_b);

    const float original_result =
        fp32DotProduct(test.a, test.b);

    const float error_vs_decoded =
        std::fabs(afp_result - fp32_result);

    const float error_vs_original =
        std::fabs(afp_result - original_result);

    std::cout << "\n[" << test.name << "]\n";

    std::cout << std::fixed << std::setprecision(8);

    std::cout << "AFP dot product:      "
              << afp_result << '\n';

    std::cout << "Decoded FP32 product: "
              << fp32_result << '\n';

    std::cout << "Original FP32 product:"
              << original_result << '\n';

    std::cout << "Error vs decoded:     "
              << error_vs_decoded << '\n';

    std::cout << "Error vs original:    "
              << error_vs_original << '\n';

    if (error_vs_decoded < 1e-5f)
    {
        std::cout << "PASS\n";
    }
    else
    {
        std::cout << "FAIL\n";
    }
}

int main()
{
    std::vector<TestCase> tests =
    {
        {
            "Basic positive values",
            {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            },
            {
                2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2
            }
        },

        {
            "All negative values",
            {
                -1, -2, -3, -4,
                -5, -6, -7, -8,
                -9, -10, -11, -12,
                -13, -14, -15, -16
            },
            {
                2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2
            }
        },

        {
            "Mixed signs",
            {
                1, -2, 3, -4,
                5, -6, 7, -8,
                9, -10, 11, -12,
                13, -14, 15, -16
            },
            {
                2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2
            }
        },

        {
            "Positive and negative B",
            {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            },
            {
                2, -2, 2, -2,
                2, -2, 2, -2,
                2, -2, 2, -2,
                2, -2, 2, -2
            }
        },

        {
            "Fractional powers of two",
            {
                1.0f, 0.5f, 0.25f, 0.125f,
                0.0625f, 0.03125f, 0.015625f, 0.0078125f,
                2.0f, 4.0f, 8.0f, 16.0f,
                32.0f, 64.0f, 128.0f, 256.0f
            },
            {
                1.0f, 2.0f, 4.0f, 8.0f,
                16.0f, 32.0f, 64.0f, 128.0f,
                0.5f, 0.25f, 0.125f, 0.0625f,
                0.03125f, 0.015625f, 0.0078125f, 0.00390625f
            }
        },

        {
            "Zeros",
            {
                0, 0, 0, 0,
                1, 2, 3, 4,
                0, 0, 0, 0,
                5, 6, 7, 8
            },
            {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            }
        },

        {
            "Large exponent differences",
            {
                256.0f, 128.0f, 64.0f, 32.0f,
                16.0f, 8.0f, 4.0f, 2.0f,
                1.0f, 0.5f, 0.25f, 0.125f,
                0.0625f, 0.03125f, 0.015625f, 0.0078125f
            },
            {
                0.0078125f, 0.015625f, 0.03125f, 0.0625f,
                0.125f, 0.25f, 0.5f, 1.0f,
                2.0f, 4.0f, 8.0f, 16.0f,
                32.0f, 64.0f, 128.0f, 256.0f
            }
        },

        {
            "Offset seven",
            {
                1.0f,
                0.5f,
                0.25f,
                0.125f,
                0.0625f,
                0.03125f,
                0.015625f,
                0.0078125f,
                1.0f,
                0.5f,
                0.25f,
                0.125f,
                0.0625f,
                0.03125f,
                0.015625f,
                0.0078125f
            },
            {
                1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f
            }
        },

        {
            "Quantization required",
            {
                1.234567f, 2.345678f, 3.456789f, 4.567891f,
                5.678912f, 6.789123f, 7.891234f, 8.912345f,
                -1.111111f, -2.222222f, -3.333333f, -4.444444f,
                -5.555555f, -6.666666f, -7.777777f, -8.888888f
            },
            {
                0.987654f, 0.876543f, 0.765432f, 0.654321f,
                0.543210f, 0.432109f, 0.321098f, 0.210987f,
                -0.123456f, -0.234567f, -0.345678f, -0.456789f,
                -0.567891f, -0.678912f, -0.789123f, -0.891234f
            }
        },

        {
            "Two blocks",
            {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,

                2, 4, 6, 8,
                10, 12, 14, 16,
                18, 20, 22, 24,
                26, 28, 30, 32
            },
            {
                2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2,

                1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1
            }
        },

        {
            "Non-multiple of block size",
            {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11
            },
            {
                2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2
            }
        }
    };

    std::cout << "AFP Arithmetic Tests\n";
    std::cout << "====================\n";

    for (const TestCase &test : tests)
    {
        runTest(test);
    }

    return 0;
}