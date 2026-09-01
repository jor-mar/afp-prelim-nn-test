#include "../src/fp16.cpp"
#include "test_utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace
{
    void testExactValues()
    {
        FP16Quantizer quantizer;

        const std::vector<float> input = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, 0.25f, 8.0f, -16.0f};

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expectNear(decoded[i], input[i], 0.0, "FP16 exact value " + std::to_string(i));
        }
    }

    void testApproximateValues()
    {
        FP16Quantizer quantizer;

        const std::vector<float> input = {1.2345f, -2.7182f, 3.14159f, 123.456f, -45.678f};

        const auto encoded =quantizer.encode(input);

        const auto decoded =quantizer.decode(encoded);

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            const float tolerance = std::max(0.001f, std::abs(input[i]) * 0.001f);

            test_utils::expectNear(decoded[i], input[i], tolerance, "FP16 approximate value " + std::to_string(i));
        }
    }

    void testStorageSize()
    {
        FP16Quantizer quantizer;

        const std::vector<float> input = {1.0f, 2.0f, 3.0f};

        const auto encoded = quantizer.encode(input);

        test_utils::expect(encoded.bitSize() == 48, "FP16 uses 16 bits per value");

        test_utils::expect(encoded.byteSize() == 6, "FP16 uses correct byte count");

        test_utils::expectNear(encoded.bitsPerValue(), 16.0, 1e-12, "FP16 bits per value is 16");
    }

    void testSpecialValues()
    {
        FP16Quantizer quantizer;

        const std::vector<float> input = {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};

        const auto encoded =quantizer.encode(input);

        const auto decoded =quantizer.decode(encoded);

        test_utils::expect(std::isinf(decoded[0]) && decoded[0] > 0, "FP16 preserves positive infinity");

        test_utils::expect(std::isinf(decoded[1]) && decoded[1] < 0, "FP16 preserves negative infinity");
    }

    void testUnderflow()
    {
        FP16Quantizer quantizer;

        const std::vector<float> input = {1.0e-10f};

        const auto encoded =quantizer.encode(input);

        const auto decoded =quantizer.decode(encoded);

        test_utils::expectNear(decoded[0], 0.0, 1e-12, "Very small FP32 value underflows to zero");
    }

    void testEmptyTensor()
    {
        FP16Quantizer quantizer;

        const std::vector<float> input;

        const auto encoded =quantizer.encode(input);

        const auto decoded =quantizer.decode(encoded);

        test_utils::expect(encoded.size() == 0, "Empty FP16 tensor has zero values");

        test_utils::expect(encoded.bitSize() == 0, "Empty FP16 tensor has zero bits");

        test_utils::expect(decoded.empty(), "Empty FP16 tensor decodes empty");
    }

}

int main()
{
    std::cout
        << "========================================\n"
        << "FP16 Tests\n"
        << "========================================\n\n";

    testExactValues();
    testApproximateValues();
    testStorageSize();
    testSpecialValues();
    testUnderflow();
    testEmptyTensor();

    return test_utils::finish();
}