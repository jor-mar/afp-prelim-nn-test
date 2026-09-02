#include "../src/afp.cpp"
#include "test_utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

namespace
{
    void testDefaultConfiguration()
    {
        AFPQuantizer quantizer;
        const AFPConfig &config = quantizer.getConfig();

        test_utils::expect(config.block_size == 16, "Default AFP8 block size is 16");
        test_utils::expect(config.exponent_bits == 8, "Default AFP8 exponent width is 8");
        test_utils::expect(config.characterization_bits == 8, "Default AFP8 characterization width is 8");
        test_utils::expect(config.offset_bits == 3, "Default AFP8 offset width is 3");
        test_utils::expect(config.mantissa_bits == 5, "Default AFP8 mantissa width is 5");
        test_utils::expect(config.enable_positive_fields, "Default AFP8 enables positive fields");
        test_utils::expect(!config.enable_zero_fields, "Default AFP8 disables zero fields");
    }

    void testFormatName()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {1.0f};
        const AFPEncodedTensor encoded = quantizer.encode(input);

        test_utils::expect(
            encoded.formatName() == "AFP8",
            "AFP format name is AFP8");
    }

    void testEmptyTensor()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input;
        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        test_utils::expect(encoded.size() == 0, "Empty AFP8 tensor has zero values");
        test_utils::expect(encoded.bitSize() == 0, "Empty AFP8 tensor has zero bits");
        test_utils::expect(encoded.blockCount() == 0, "Empty AFP8 tensor has zero blocks");
        test_utils::expect(decoded.empty(), "Empty AFP8 tensor decodes to an empty vector");
    }

    void testSingleFullBlockBitSize()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f};

        const AFPEncodedTensor encoded = quantizer.encode(input);

        test_utils::expect(encoded.blockCount() == 1, "16 values produce one AFP8 block");
        test_utils::expect(encoded.bitSize() == 160, "One AFP8 block occupies 160 bits");
        test_utils::expectNear(encoded.bitsPerValue(), 10.0, 1e-12, "AFP8 uses 10 bits per value for a full block");
    }

    void testPartialBlockPadding()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        test_utils::expect(encoded.blockCount() == 1, "Partial AFP8 block still occupies one physical block");
        test_utils::expect(encoded.bitSize() == 160, "Partial AFP8 block is padded to 160 bits");
        test_utils::expect(decoded.size() == input.size(), "AFP8 removes padding after decoding");

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expectNear(
                decoded[i],
                input[i],
                0.3,
                "Partial block preserves value " + std::to_string(i));
        }
    }

    void testMultipleBlocks()
    {
        AFPQuantizer quantizer;

        std::vector<float> input;

        for (int i = 0; i < 33; ++i)
            input.push_back(static_cast<float>(i + 1));

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        test_utils::expect(encoded.blockCount() == 3, "33 values produce three AFP8 blocks");
        test_utils::expect(encoded.bitSize() == 480, "Three AFP8 blocks occupy 480 bits");
        test_utils::expect(decoded.size() == input.size(), "Multiple AFP8 blocks preserve value count");
    }

    void testExactPowersOfTwo()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            1.0f,
            2.0f,
            4.0f,
            8.0f,
            16.0f,
            32.0f,
            64.0f,
            128.0f};

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expectNear(
                decoded[i],
                input[i],
                0.0,
                "AFP8 exactly reconstructs power of two " + std::to_string(i));
        }
    }

    void testNegativePowersOfTwo()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            -1.0f,
            -2.0f,
            -4.0f,
            -8.0f,
            -16.0f,
            -32.0f,
            -64.0f,
            -128.0f};

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expect(decoded[i] < 0.0f, "AFP8 preserves negative sign " + std::to_string(i));

            test_utils::expectNear(
                decoded[i],
                input[i],
                0.0,
                "AFP8 exactly reconstructs negative power of two " + std::to_string(i));
        }
    }

    void testMixedSignBlock()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            -1.0f, 1.0f, -2.0f, 2.0f,
            -4.0f, 4.0f, -8.0f, 8.0f,
            -0.5f, 0.5f, -0.25f, 0.25f,
            -0.125f, 0.125f, -0.0625f, 0.0625f};

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expect(
                std::signbit(decoded[i]) == std::signbit(input[i]),
                "AFP8 preserves mixed sign " + std::to_string(i));

            test_utils::expectNear(
                decoded[i],
                input[i],
                0.001,
                "AFP8 reconstructs mixed-sign value " + std::to_string(i));
        }
    }

    void testFirstHalfPositiveField()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            1.03125f, 1.0625f, 1.09375f, 1.125f,
            1.15625f, 1.1875f, 1.21875f, 1.25f,

            -1.03125f, 1.0625f, -1.09375f, 1.125f,
            -1.15625f, 1.1875f, -1.21875f, 1.25f};

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < 8; ++i)
        {
            test_utils::expectNear(
                decoded[i],
                input[i],
                0.001,
                "AFP8 positive first half retains extra precision " + std::to_string(i));
        }

        for (std::size_t i = 8; i < input.size(); ++i)
        {
            test_utils::expect(
                std::signbit(decoded[i]) == std::signbit(input[i]),
                "AFP8 mixed second half preserves sign " + std::to_string(i));
        }
    }

    void testSecondHalfPositiveField()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            -1.0f, 1.0f, -1.25f, 1.25f,
            -1.5f, 1.5f, -1.75f, 1.75f,

            1.03125f, 1.0625f, 1.09375f, 1.125f,
            1.15625f, 1.1875f, 1.21875f, 1.25f};

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < 8; ++i)
        {
            test_utils::expect(
                std::signbit(decoded[i]) == std::signbit(input[i]),
                "AFP8 mixed first half preserves sign " + std::to_string(i));
        }

        for (std::size_t i = 8; i < input.size(); ++i)
        {
            test_utils::expectNear(
                decoded[i],
                input[i],
                0.001,
                "AFP8 positive second half retains extra precision " + std::to_string(i));
        }
    }

    void testPositiveFieldPrecision()
    {
        AFPQuantizer quantizer;

        const std::vector<float> positive_input = {
            1.015625f, 1.046875f, 1.078125f, 1.109375f,
            1.140625f, 1.171875f, 1.203125f, 1.234375f,

            1.265625f, 1.296875f, 1.328125f, 1.359375f,
            1.390625f, 1.421875f, 1.453125f, 1.484375f};

        const AFPEncodedTensor encoded = quantizer.encode(positive_input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < positive_input.size(); ++i)
        {
            test_utils::expectNear(
                decoded[i],
                positive_input[i],
                0.001,
                "AFP8 positive field preserves 6-bit mantissa step " + std::to_string(i));
        }
    }

    void testZeroEncoding()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            0.0f, 1.0f, 0.0f, -1.0f,
            0.0f, 0.5f, 0.0f, -0.5f,
            0.0f, 2.0f, 0.0f, -2.0f,
            0.0f, 4.0f, 0.0f, -4.0f};

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < input.size(); i += 2)
        {
            test_utils::expectNear(
                decoded[i],
                0.0,
                0.0,
                "AFP8 exactly reconstructs zero " + std::to_string(i));
        }
    }

    void testAllZeroBlock()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input(16, 0.0f);
        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        test_utils::expect(encoded.blockCount() == 1, "All-zero AFP8 tensor has one block");
        test_utils::expect(encoded.bitSize() == 160, "All-zero AFP8 block occupies 160 bits");

        for (std::size_t i = 0; i < decoded.size(); ++i)
        {
            test_utils::expectNear(
                decoded[i],
                0.0,
                0.0,
                "AFP8 all-zero value " + std::to_string(i));
        }
    }

    void testOffsetsZeroThroughSix()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            64.0f,
            32.0f,
            16.0f,
            8.0f,
            4.0f,
            2.0f,
            1.0f,

            64.0f,
            32.0f,
            16.0f,
            8.0f,
            4.0f,
            2.0f,
            1.0f,
            0.5f};

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < input.size() - 1; ++i)
        {
            test_utils::expectNear(
                decoded[i],
                input[i],
                0.0,
                "AFP8 reconstructs normal offset value " + std::to_string(i));
        }
    }

    void testOffsetSevenBoundary()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            1.0f,
            0.5f,
            0.25f,
            0.125f,
            0.0625f,
            0.03125f,
            0.015625f,
            0.0078125f,

            0.00390625f,
            0.001953125f,
            0.0009765625f,
            0.00048828125f,
            0.000244140625f,
            0.0001220703125f,
            0.00006103515625f,
            0.000030517578125f};

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < 7; ++i)
        {
            test_utils::expectNear(
                decoded[i],
                input[i],
                0.0,
                "AFP8 normal offset before boundary " + std::to_string(i));
        }

        test_utils::expect(
            decoded[7] >= 0.0f,
            "AFP8 offset-7 boundary remains non-negative");

        for (std::size_t i = 8; i < input.size(); ++i)
        {
            test_utils::expect(
                std::isfinite(decoded[i]),
                "AFP8 denormal value remains finite " + std::to_string(i));

            test_utils::expect(
                decoded[i] >= 0.0f,
                "AFP8 denormal value remains non-negative " + std::to_string(i));
        }
    }

    void testVerySmallValues()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            1.0f,
            std::ldexp(1.0f, -8),
            std::ldexp(1.0f, -9),
            std::ldexp(1.0f, -10),
            std::ldexp(1.0f, -11),
            std::ldexp(1.0f, -12),
            std::ldexp(1.0f, -13),
            std::ldexp(1.0f, -14)};

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        test_utils::expectNear(decoded[0], 1.0, 0.0, "AFP8 preserves shared maximum value");

        for (std::size_t i = 1; i < input.size(); ++i)
        {
            test_utils::expect(
                std::isfinite(decoded[i]),
                "AFP8 very small value remains finite " + std::to_string(i));

            test_utils::expect(
                decoded[i] >= 0.0f,
                "AFP8 very small value remains non-negative " + std::to_string(i));
        }
    }

    void testNearestRounding()
    {
        AFPQuantizer quantizer;

        const std::vector<float> input = {
            1.001f, 1.012f, 1.024f, 1.038f,
            1.051f, 1.067f, 1.081f, 1.099f,
            1.113f, 1.129f, 1.147f, 1.163f,
            1.181f, 1.199f, 1.217f, 1.239f};

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            test_utils::expect(
                std::isfinite(decoded[i]),
                "AFP8 rounded value remains finite " + std::to_string(i));

            test_utils::expect(
                decoded[i] > 0.0f,
                "AFP8 rounded value remains positive " + std::to_string(i));

            test_utils::expectNear(
                decoded[i],
                input[i],
                0.02,
                "AFP8 nearest rounding remains bounded " + std::to_string(i));
        }
    }

    void testRandomizedRoundTrip()
    {
        AFPQuantizer quantizer;

        std::mt19937 generator(42);
        std::uniform_real_distribution<float> distribution(-100.0f, 100.0f);

        std::vector<float> input;
        input.reserve(1000);

        for (int i = 0; i < 1000; ++i)
            input.push_back(distribution(generator));

        const AFPEncodedTensor encoded = quantizer.encode(input);
        const std::vector<float> decoded = quantizer.decode(encoded);

        test_utils::expect(
            decoded.size() == input.size(),
            "AFP8 randomized test preserves value count");

        double total_error = 0.0;
        double maximum_error = 0.0;

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            const double error = std::abs(
                static_cast<double>(input[i]) -
                static_cast<double>(decoded[i]));

            total_error += error;
            maximum_error = std::max(maximum_error, error);

            test_utils::expect(
                std::isfinite(decoded[i]),
                "AFP8 randomized value remains finite " + std::to_string(i));
        }

        const double mean_error = total_error / input.size();

        test_utils::expect(
            mean_error < 1.0,
            "AFP8 randomized mean error remains bounded");

        test_utils::expect(
            maximum_error < 8.0,
            "AFP8 randomized maximum error remains bounded");
    }

    void testInvalidConfigurations()
    {
        test_utils::expectThrows(
            "AFP8 rejects block size other than 16",
            []()
            {
                AFPConfig config;
                config.block_size = 8;

                AFPQuantizer quantizer(config);
            });

        test_utils::expectThrows(
            "AFP8 rejects exponent width other than 8",
            []()
            {
                AFPConfig config;
                config.exponent_bits = 7;

                AFPQuantizer quantizer(config);
            });

        test_utils::expectThrows(
            "AFP8 rejects characterization width other than 8",
            []()
            {
                AFPConfig config;
                config.characterization_bits = 7;

                AFPQuantizer quantizer(config);
            });

        test_utils::expectThrows(
            "AFP8 rejects offset width other than 3",
            []()
            {
                AFPConfig config;
                config.offset_bits = 2;

                AFPQuantizer quantizer(config);
            });

        test_utils::expectThrows(
            "AFP8 rejects mantissa width other than 5",
            []()
            {
                AFPConfig config;
                config.mantissa_bits = 6;

                AFPQuantizer quantizer(config);
            });

        test_utils::expectThrows(
            "AFP8 requires positive fields",
            []()
            {
                AFPConfig config;
                config.enable_positive_fields = false;

                AFPQuantizer quantizer(config);
            });

        test_utils::expectThrows(
            "AFP8 rejects zero fields",
            []()
            {
                AFPConfig config;
                config.enable_zero_fields = true;

                AFPQuantizer quantizer(config);
            });
    }

    void testSpecialValuesRejected()
    {
        AFPQuantizer quantizer;

        test_utils::expectThrows(
            "AFP8 rejects positive infinity",
            [&quantizer]()
            {
                quantizer.encode({1.0f,
                                  std::numeric_limits<float>::infinity()});
            });

        test_utils::expectThrows(
            "AFP8 rejects negative infinity",
            [&quantizer]()
            {
                quantizer.encode({-std::numeric_limits<float>::infinity()});
            });

        test_utils::expectThrows(
            "AFP8 rejects NaN",
            [&quantizer]()
            {
                quantizer.encode({1.0f,
                                  std::numeric_limits<float>::quiet_NaN()});
            });
    }

}

int main()
{
    std::cout << "========================================\n";
    std::cout << "AFP8 Tests\n";
    std::cout << "========================================\n\n";

    testDefaultConfiguration();
    testFormatName();

    testEmptyTensor();

    testSingleFullBlockBitSize();
    testPartialBlockPadding();
    testMultipleBlocks();

    testExactPowersOfTwo();
    testNegativePowersOfTwo();
    testMixedSignBlock();

    testFirstHalfPositiveField();
    testSecondHalfPositiveField();
    testPositiveFieldPrecision();

    testZeroEncoding();
    testAllZeroBlock();

    testOffsetsZeroThroughSix();
    testOffsetSevenBoundary();
    testVerySmallValues();

    testNearestRounding();
    testRandomizedRoundTrip();

    testInvalidConfigurations();
    testSpecialValuesRejected();

    return test_utils::finish();
}