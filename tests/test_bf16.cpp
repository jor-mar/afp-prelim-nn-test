#include "../src/bf16.cpp"
#include "test_utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace
{
    void testBasicRoundTrip()
    {
        BF16Quantizer quantizer;

        const std::vector<float> input = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 10.5f, -10.5f};

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        test_utils::expect(decoded.size() == input.size(), "BF16 preserves value count");

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expectNear(decoded[i], input[i], 0.05, "BF16 basic value " + std::to_string(i));
        }
    }

    void testExactValues()
    {
        BF16Quantizer quantizer;

        const std::vector<float> input = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 0.25f, -8.0f};

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expectNear(decoded[i], input[i], 0.0, "BF16 exact power-of-two value " + std::to_string(i));
        }
    }

    void testStorageSize()
    {
        BF16Quantizer quantizer;

        const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};

        const auto encoded = quantizer.encode(input);

        test_utils::expect(encoded.bitSize() == 64, "BF16 uses 16 bits per value");

        test_utils::expect(encoded.byteSize() == 8, "BF16 uses correct byte count");

        test_utils::expectNear(encoded.bitsPerValue(), 16.0, 1e-12, "BF16 bits per value is 16");
    }

    void testSpecialValues()
    {
        BF16Quantizer quantizer;

        const std::vector<float> input = {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        test_utils::expect(std::isinf(decoded[0]) && decoded[0] > 0, "BF16 preserves positive infinity");

        test_utils::expect(std::isinf(decoded[1]) && decoded[1] < 0, "BF16 preserves negative infinity");
    }

    void testEmptyTensor()
    {
        BF16Quantizer quantizer;

        const std::vector<float> input;

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        test_utils::expect(encoded.size() == 0, "Empty BF16 tensor has zero values");

        test_utils::expect(encoded.bitSize() == 0, "Empty BF16 tensor has zero bits");

        test_utils::expect(decoded.empty(), "Empty BF16 tensor decodes empty");
    }

}

int main()
{
    std::cout
        << "========================================\n"
        << "BF16 Tests\n"
        << "========================================\n\n";

    testBasicRoundTrip();
    testExactValues();
    testStorageSize();
    testSpecialValues();
    testEmptyTensor();

    return test_utils::finish();
}
