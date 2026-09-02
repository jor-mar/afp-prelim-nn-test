#include "../include/afp_math.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace
{
constexpr std::size_t shared_exponent_bits = 8;
constexpr std::size_t characterization_bits = 8;

constexpr std::size_t sign_bits = 1;
constexpr std::size_t offset_bits = 3;
constexpr std::size_t mantissa_bits = 5;

constexpr std::size_t value_bits =
    sign_bits + offset_bits + mantissa_bits;

constexpr std::size_t block_header_bits =
    shared_exponent_bits + characterization_bits;

constexpr std::size_t first_half_positive_bit = 0;
constexpr std::size_t second_half_positive_bit = 1;

constexpr int maximum_offset = 7;

int decodeSharedExponent(std::uint8_t exponent)
{
    if (exponent == 0)
    {
        return -126;
    }

    return static_cast<int>(exponent) - 127;
}
}

std::uint64_t AFPArithmetic::readBits(
    const AFPEncodedTensor &tensor,
    std::size_t bit_offset,
    std::size_t bit_count)
{
    return tensor.bits_.readBits(
        bit_offset,
        bit_count);
}

std::int64_t AFPArithmetic::dotProductBlock(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b,
    std::size_t block)
{
    const std::size_t a_base =
        a.block_offsets_[block];

    const std::size_t b_base =
        b.block_offsets_[block];

    const std::uint8_t a_characterization =
        static_cast<std::uint8_t>(
            readBits(
                a,
                a_base + shared_exponent_bits,
                characterization_bits));

    const std::uint8_t b_characterization =
        static_cast<std::uint8_t>(
            readBits(
                b,
                b_base + shared_exponent_bits,
                characterization_bits));

    struct Product
    {
        std::int64_t value;
        int shift;
    };

    Product products[16];

    int minimum_shift = 0;
    bool found_product = false;

    for (std::size_t i = 0; i < 16; ++i)
    {
        const std::size_t a_value_base =
            a_base +
            block_header_bits +
            i * value_bits;

        const std::size_t b_value_base =
            b_base +
            block_header_bits +
            i * value_bits;

        const std::uint8_t a_first_field =
            static_cast<std::uint8_t>(
                readBits(
                    a,
                    a_value_base,
                    sign_bits));

        const std::uint8_t b_first_field =
            static_cast<std::uint8_t>(
                readBits(
                    b,
                    b_value_base,
                    sign_bits));

        const std::uint8_t a_offset =
            static_cast<std::uint8_t>(
                readBits(
                    a,
                    a_value_base + sign_bits,
                    offset_bits));

        const std::uint8_t b_offset =
            static_cast<std::uint8_t>(
                readBits(
                    b,
                    b_value_base + sign_bits,
                    offset_bits));

        std::uint8_t a_mantissa =
            static_cast<std::uint8_t>(
                readBits(
                    a,
                    a_value_base +
                        sign_bits +
                        offset_bits,
                    mantissa_bits));

        std::uint8_t b_mantissa =
            static_cast<std::uint8_t>(
                readBits(
                    b,
                    b_value_base +
                        sign_bits +
                        offset_bits,
                    mantissa_bits));

        bool a_positive;
        bool b_positive;

        if (i < 8)
        {
            a_positive =
                ((a_characterization >>
                  first_half_positive_bit) & 1U) != 0;

            b_positive =
                ((b_characterization >>
                  first_half_positive_bit) & 1U) != 0;
        }
        else
        {
            a_positive =
                ((a_characterization >>
                  second_half_positive_bit) & 1U) != 0;

            b_positive =
                ((b_characterization >>
                  second_half_positive_bit) & 1U) != 0;
        }

        std::uint8_t a_sign = 0;
        std::uint8_t b_sign = 0;

        if (a_positive)
        {
            a_mantissa |=
                static_cast<std::uint8_t>(
                    a_first_field << 5);
        }
        else
        {
            a_sign = a_first_field;
        }

        if (b_positive)
        {
            b_mantissa |=
                static_cast<std::uint8_t>(
                    b_first_field << 5);
        }
        else
        {
            b_sign = b_first_field;
        }

        if (a_offset == maximum_offset &&
            a_mantissa == 0)
        {
            products[i] = {0, 0};
            continue;
        }

        if (b_offset == maximum_offset &&
            b_mantissa == 0)
        {
            products[i] = {0, 0};
            continue;
        }

        std::int64_t a_significand;
        std::int64_t b_significand;

        int a_fraction_bits;
        int b_fraction_bits;

        if (a_offset < maximum_offset)
        {
            if (a_positive)
            {
                a_significand =
                    64 + a_mantissa;

                a_fraction_bits = 6;
            }
            else
            {
                a_significand =
                    32 + a_mantissa;

                a_fraction_bits = 5;
            }
        }
        else
        {
            a_significand =
                a_mantissa;

            a_fraction_bits =
                a_positive ? 6 : 5;
        }

        if (b_offset < maximum_offset)
        {
            if (b_positive)
            {
                b_significand =
                    64 + b_mantissa;

                b_fraction_bits = 6;
            }
            else
            {
                b_significand =
                    32 + b_mantissa;

                b_fraction_bits = 5;
            }
        }
        else
        {
            b_significand =
                b_mantissa;

            b_fraction_bits =
                b_positive ? 6 : 5;
        }

        std::int64_t product =
            a_significand *
            b_significand;

        if ((a_sign ^ b_sign) != 0)
        {
            product = -product;
        }

        /*
         * The product represents:
         *
         * product *
         * 2^(-(a_offset + b_offset)
         *    -(a_fraction_bits + b_fraction_bits))
         *
         * The exponent shared by the block is handled
         * after the fixed-point accumulation.
         */
        const int product_shift =
            static_cast<int>(a_offset) +
            static_cast<int>(b_offset) +
            a_fraction_bits +
            b_fraction_bits;

        products[i] =
        {
            product,
            product_shift
        };

        if (!found_product ||
            product_shift < minimum_shift)
        {
            minimum_shift = product_shift;
            found_product = true;
        }
    }

    if (!found_product)
    {
        return 0;
    }

    std::int64_t accumulator = 0;

    for (std::size_t i = 0; i < 16; ++i)
    {
        if (products[i].value == 0)
        {
            continue;
        }

        const int shift =
            products[i].shift -
            minimum_shift;

        std::int64_t aligned_product;

        if (shift > 0)
        {
            aligned_product =
                products[i].value >> shift;
        }
        else if (shift < 0)
        {
            aligned_product =
                products[i].value << (-shift);
        }
        else
        {
            aligned_product =
                products[i].value;
        }

        accumulator += aligned_product;
    }

    /*
     * The accumulator represents:
     *
     * accumulator *
     * 2^(-minimum_shift)
     *
     * The shared exponents are applied by dotProduct().
     */
    return accumulator;
}

float AFPArithmetic::dotProduct(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument(
            "AFP dot product requires tensors of equal size");
    }

    if (a.size() == 0)
    {
        return 0.0f;
    }

    if (a.blockSize() != b.blockSize())
    {
        throw std::invalid_argument(
            "AFP dot product requires equal block sizes");
    }

    if (a.blockSize() != 16)
    {
        throw std::invalid_argument(
            "AFP arithmetic currently requires block size 16");
    }

    if (a.size() % 16 != 0)
    {
        throw std::invalid_argument(
            "AFP dot product requires block-aligned tensors");
    }

    double result = 0.0;

    for (std::size_t block = 0;
         block < a.blockCount();
         ++block)
    {
        const std::size_t a_base =
            a.block_offsets_[block];

        const std::size_t b_base =
            b.block_offsets_[block];

        const std::uint8_t a_exponent_field =
            static_cast<std::uint8_t>(
                readBits(
                    a,
                    a_base,
                    shared_exponent_bits));

        const std::uint8_t b_exponent_field =
            static_cast<std::uint8_t>(
                readBits(
                    b,
                    b_base,
                    shared_exponent_bits));

        const int a_exponent =
            decodeSharedExponent(a_exponent_field);

        const int b_exponent =
            decodeSharedExponent(b_exponent_field);

        const std::int64_t accumulator =
            dotProductBlock(
                a,
                b,
                block);

        /*
         * dotProductBlock() currently returns an accumulator
         * whose fractional scale depends on the smallest
         * product shift in the block.
         *
         * Reconstructing that scale requires determining
         * the minimum product shift again.
         */
        const std::uint8_t a_characterization =
            static_cast<std::uint8_t>(
                readBits(
                    a,
                    a_base + shared_exponent_bits,
                    characterization_bits));

        const std::uint8_t b_characterization =
            static_cast<std::uint8_t>(
                readBits(
                    b,
                    b_base + shared_exponent_bits,
                    characterization_bits));

        int minimum_shift = 0;
        bool found_product = false;

        for (std::size_t i = 0; i < 16; ++i)
        {
            const std::size_t a_value_base =
                a_base +
                block_header_bits +
                i * value_bits;

            const std::size_t b_value_base =
                b_base +
                block_header_bits +
                i * value_bits;

            const std::uint8_t a_first_field =
                static_cast<std::uint8_t>(
                    readBits(
                        a,
                        a_value_base,
                        sign_bits));

            const std::uint8_t b_first_field =
                static_cast<std::uint8_t>(
                    readBits(
                        b,
                        b_value_base,
                        sign_bits));

            const std::uint8_t a_offset =
                static_cast<std::uint8_t>(
                    readBits(
                        a,
                        a_value_base + sign_bits,
                        offset_bits));

            const std::uint8_t b_offset =
                static_cast<std::uint8_t>(
                    readBits(
                        b,
                        b_value_base + sign_bits,
                        offset_bits));

            std::uint8_t a_mantissa =
                static_cast<std::uint8_t>(
                    readBits(
                        a,
                        a_value_base +
                            sign_bits +
                            offset_bits,
                        mantissa_bits));

            std::uint8_t b_mantissa =
                static_cast<std::uint8_t>(
                    readBits(
                        b,
                        b_value_base +
                            sign_bits +
                            offset_bits,
                        mantissa_bits));

            bool a_positive;
            bool b_positive;

            if (i < 8)
            {
                a_positive =
                    ((a_characterization >>
                      first_half_positive_bit) & 1U) != 0;

                b_positive =
                    ((b_characterization >>
                      first_half_positive_bit) & 1U) != 0;
            }
            else
            {
                a_positive =
                    ((a_characterization >>
                      second_half_positive_bit) & 1U) != 0;

                b_positive =
                    ((b_characterization >>
                      second_half_positive_bit) & 1U) != 0;
            }

            if (a_positive)
            {
                a_mantissa |=
                    static_cast<std::uint8_t>(
                        a_first_field << 5);
            }

            if (b_positive)
            {
                b_mantissa |=
                    static_cast<std::uint8_t>(
                        b_first_field << 5);
            }

            if (a_offset == maximum_offset &&
                a_mantissa == 0)
            {
                continue;
            }

            if (b_offset == maximum_offset &&
                b_mantissa == 0)
            {
                continue;
            }

            const int a_fraction_bits =
                a_positive ? 6 : 5;

            const int b_fraction_bits =
                b_positive ? 6 : 5;

            const int product_shift =
                static_cast<int>(a_offset) +
                static_cast<int>(b_offset) +
                a_fraction_bits +
                b_fraction_bits;

            if (!found_product ||
                product_shift < minimum_shift)
            {
                minimum_shift = product_shift;
                found_product = true;
            }
        }

        if (!found_product)
        {
            continue;
        }

        result +=
            std::ldexp(
                static_cast<double>(accumulator),
                a_exponent +
                b_exponent -
                minimum_shift);
    }

    return static_cast<float>(result);
}