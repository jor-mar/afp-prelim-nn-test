#include "../src/afp.cpp"
#include "test_utils.hpp"

#include <cmath>
#include <vector>

namespace
{
    void testBasicRoundTrip()
    {
        AFPConfig config;

        config.block_size = 4;
        config.offset_bits = 3;
        config.mantissa_bits = 8;
        config.exponent_bits = 8;

        AFPQuantizer quantizer(config);

        const std::vector<float> input = {1.0f, 0.5f, -0.25f, 0.125f};

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        test_utils::expect(decoded.size() == input.size(), "AFP preserves value count");

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expect(
                decoded[i] == 0.0f ||
                    (decoded[i] > 0.0f) ==
                        (input[i] > 0.0f),
                "AFP preserves sign " +
                    std::to_string(i));

            test_utils::expectNear(
                decoded[i],
                input[i],
                0.02,
                "AFP reconstructs value " +
                    std::to_string(i));
        }
    }

    void testMultipleBlocks()
    {
        AFPConfig config;

        config.block_size = 3;
        config.offset_bits = 4;
        config.mantissa_bits = 10;
        config.exponent_bits = 8;

        AFPQuantizer quantizer(config);

        const std::vector<float> input = {
            1.0f,
            0.5f,
            -0.25f,

            8.0f,
            4.0f,
            -2.0f,

            0.125f,
            -0.0625f};

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        test_utils::expect(decoded.size() == input.size(), "AFP multiple blocks preserve value count");

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expectNear(
                decoded[i],
                input[i],
                0.02,
                "AFP multiple block value " +
                    std::to_string(i));
        }
    }

    void testZeroValues()
    {
        AFPConfig config;

        config.block_size = 4;
        config.mantissa_bits = 8;

        AFPQuantizer quantizer(config);

        const std::vector<float> input = {1.0f, 0.0f, -0.5f, 0.0f};

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        test_utils::expectNear(
            decoded[1],
            0.0,
            0.0,
            "AFP preserves first zero");

        test_utils::expectNear(
            decoded[3],
            0.0,
            0.0,
            "AFP preserves second zero");
    }

    void testZeroTensor()
    {
        AFPConfig config;

        config.block_size = 4;

        AFPQuantizer quantizer(config);

        const std::vector<float> input = {
            0.0f,
            0.0f,
            0.0f,
            0.0f};

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        for (float value : decoded)
        {
            test_utils::expectNear(
                value,
                0.0,
                0.0,
                "AFP zero tensor decodes to zero");
        }
    }

    void testPartialBlock()
    {
        AFPConfig config;

        config.block_size = 4;
        config.offset_bits = 3;
        config.mantissa_bits = 8;

        AFPQuantizer quantizer(config);

        const std::vector<float> input = {
            1.0f,
            0.5f,
            0.25f,

            8.0f,
            4.0f};

        const auto encoded = quantizer.encode(input);

        const auto decoded = quantizer.decode(encoded);

        test_utils::expect(decoded.size() == input.size(), "AFP partial block preserves value count");

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expectNear(
                decoded[i],
                input[i],
                0.05,
                "AFP partial block value " +
                    std::to_string(i));
        }
    }

    void testConfigurationValidation()
    {
        test_utils::expectThrows(
            "AFP rejects zero block size",
            []()
            {
                AFPConfig config;

                config.block_size = 0;

                AFPQuantizer quantizer(config);
            });

        test_utils::expectThrows(
            "AFP rejects zero offset bits",
            []()
            {
                AFPConfig config;

                config.offset_bits = 0;

                AFPQuantizer quantizer(config);
            });

        test_utils::expectThrows(
            "AFP rejects zero mantissa bits",
            []()
            {
                AFPConfig config;

                config.mantissa_bits = 0;

                AFPQuantizer quantizer(config);
            });
    }

    void testCompressionSize()
    {
        AFPConfig config;

        config.block_size = 4;
        config.offset_bits = 3;
        config.mantissa_bits = 5;
        config.exponent_bits = 8;

        AFPQuantizer quantizer(config);

        const std::vector<float> input = {
            1.0f,
            2.0f,
            3.0f,
            4.0f};

        const auto encoded = quantizer.encode(input);

        test_utils::expect(encoded.bitSize() == 44, "AFP encoded bit size is correct");
    }

}

int main()
{
    std::cout
        << "========================================\n"
        << "AFP Tests\n"
        << "========================================\n\n";

    testBasicRoundTrip();
    testMultipleBlocks();
    testZeroValues();
    testZeroTensor();
    testPartialBlock();
    testConfigurationValidation();
    testCompressionSize();

    return test_utils::finish();
}
