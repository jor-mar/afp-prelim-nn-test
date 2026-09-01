#include "../src/int8.cpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
    void testBasicRoundTrip()
    {
        INT8Quantizer quantizer;

        const std::vector<float> input = {-10.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 10.0f};

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        const float tolerance = encoded.scale() / 2.0f + 1e-6f;

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expectNear(decoded[i], input[i], tolerance, "INT8 round-trip value " + std::to_string(i));
        }
    }

    void testStorageSize()
    {
        INT8Quantizer quantizer;

        const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};

        const auto encoded = quantizer.encode(input);

        test_utils::expect(encoded.bitSize() == 32, "INT8 uses 8 bits per value");

        test_utils::expect(encoded.byteSize() == 4, "INT8 uses correct byte count");

        test_utils::expectNear(encoded.bitsPerValue(), 8.0, 1e-12, "INT8 bits per value is 8");
    }

    void testZeroTensor()
    {
        INT8Quantizer quantizer;

        const std::vector<float> input = {0.0f, 0.0f, 0.0f};

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        test_utils::expectNear(encoded.scale(), 1.0, 0.0, "Zero tensor uses scale 1");

        for (float value : decoded)
        {
            test_utils::expectNear(value, 0.0, 0.0, "Zero tensor decodes to zero");
        }
    }

    void testEmptyTensor()
    {
        INT8Quantizer quantizer;

        const std::vector<float> input;

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        test_utils::expect(encoded.size() == 0, "Empty INT8 tensor has zero values");

        test_utils::expect(encoded.bitSize() == 0, "Empty INT8 tensor has zero bits");

        test_utils::expect(decoded.empty(), "Empty INT8 tensor decodes empty");
    }

    void testScale()
    {
        INT8Quantizer quantizer;

        const std::vector<float> input = {-127.0f, 0.0f, 127.0f};

        const auto encoded = quantizer.encode(input);

        test_utils::expectNear(encoded.scale(), 1.0, 1e-6, "INT8 computes expected scale");
    }

}

int main()
{
    std::cout
        << "========================================\n"
        << "INT8 Tests\n"
        << "========================================\n\n";

    testBasicRoundTrip();
    testStorageSize();
    testZeroTensor();
    testEmptyTensor();
    testScale();

    return test_utils::finish();
}
