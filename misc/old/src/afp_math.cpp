#include "../include/afp_math.hpp"
#include "../include/afp_encoded_tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Constants
constexpr std::size_t AFPArithmetic::block_size;
constexpr std::size_t AFPArithmetic::half_block_size;
constexpr int AFPArithmetic::maximum_offset;
constexpr int AFPArithmetic::fp32_exponent_bias;

// ============================================================================
// Internal AFP Value Helpers
// ============================================================================

/*
namespace
{
    bool isZeroAFPValue(const AFPArithmetic::AFPValue &value)
    {
        return value.offset == AFPArithmetic::maximum_offset &&
               value.mantissa == 0;
    }

    int mantissaBitsForValue(const AFPArithmetic::AFPValue &value)
    {
        /*
         * Normal fields have an implicit leading 1.
         *
         * 5-bit mantissa:
         *     implicit + 5 stored bits
         *     => mantissa range 32..63
         *
         * Positive fields reclaim the sign bit and therefore have
         * a 6-bit mantissa.
         *
         * 6-bit mantissa:
         *     implicit + 6 stored bits
         *     => mantissa range 64..127
         *
         * For offset == 7 there is no implicit leading 1:
         *
         *     5-bit field => 0..31
         *     6-bit field => 0..63
         *
         * Therefore values >= 32 at offset 7 must have come from
         * a positive 6-bit field.
         //
        if (value.offset < AFPArithmetic::maximum_offset)
        {
            return value.mantissa >= 64 ? 6 : 5;
        }

        return value.mantissa >= 32 ? 6 : 5;
    }
}
*/

bool AFPArithmetic::isZeroAFPValue(const AFPValue &value)
{
    return value.offset == maximum_offset &&
           value.mantissa == 0;
}

int AFPArithmetic::mantissaBitsForValue(const AFPValue &value)
{
    if (value.offset < maximum_offset)
    {
        if (value.mantissa >= 64)
            return 6;

        return 5;
    }

    if (value.mantissa >= 32)
        return 6;

    return 5;
}

// ============================================================================
// Utility Functions
// ============================================================================

int AFPArithmetic::integerLog2(std::uint64_t value)
{
    if (value == 0)
        return -1;

    int result = -1;

    while (value != 0)
    {
        value >>= 1;
        ++result;
    }

    return result;
}

std::uint64_t AFPArithmetic::shiftRightRounded(
    std::uint64_t value,
    int shift)
{
    if (shift <= 0)
        return value;

    if (shift >= 64)
        return 0;

    const std::uint64_t truncated =
        value >> shift;

    const std::uint64_t remainder =
        value & maskBits(shift);

    const std::uint64_t halfway =
        std::uint64_t{1} << (shift - 1);

    if (remainder > halfway)
        return truncated + 1;

    if (remainder == halfway &&
        (truncated & 1U))
    {
        return truncated + 1;
    }

    return truncated;
}

std::uint64_t AFPArithmetic::maskBits(int bit_count)
{
    if (bit_count <= 0)
        return 0;

    if (bit_count >= 64)
        return std::numeric_limits<std::uint64_t>::max();

    return (std::uint64_t{1} << bit_count) - 1;
}

int AFPArithmetic::decodeSharedExponent(
    std::uint8_t encoded)
{
    if (encoded == 0)
        return -126;

    return static_cast<int>(encoded) -
           fp32_exponent_bias;
}

std::uint8_t AFPArithmetic::encodeSharedExponent(
    int exponent)
{
    if (exponent < -126)
        exponent = -126;

    if (exponent > 127)
        exponent = 127;

    return static_cast<std::uint8_t>(
        exponent + fp32_exponent_bias
    );
}

void AFPArithmetic::validateCompatible(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    if (a.size() != b.size())
        throw std::invalid_argument(
            "AFP tensor size mismatch");

    if (a.config_.block_size != b.config_.block_size)
        throw std::invalid_argument(
            "AFP block size mismatch");
}

void AFPArithmetic::normalizeAccumulator(
    AFPAccumulator &acc)
{
    if (acc.zero ||
        acc.significand == 0)
    {
        acc.zero = true;
        acc.negative = false;
        acc.significand = 0;
        acc.exponent = 0;

        return;
    }

    const std::uint64_t magnitude =
        static_cast<std::uint64_t>(
            acc.significand);

    constexpr int accumulator_bits = 48;

    const int highest_bit =
        integerLog2(magnitude);

    if (highest_bit >= accumulator_bits)
    {
        const int shift =
            highest_bit -
            accumulator_bits +
            1;

        acc.significand =
            static_cast<std::int64_t>(
                shiftRightRounded(
                    magnitude,
                    shift));

        acc.exponent += shift;
    }
}

// ============================================================================
// AFP Value Reading/Writing
// ============================================================================

AFPArithmetic::AFPValue AFPArithmetic::readAFPValue(
    const AFPEncodedTensor &tensor,
    std::size_t block_index,
    std::size_t value_index)
{
    constexpr std::size_t shared_exponent_bits = 8;
    constexpr std::size_t characterization_bits = 8;
    constexpr std::size_t first_field_bits = 1;
    constexpr std::size_t offset_bits = 3;
    constexpr std::size_t stored_mantissa_bits = 5;

    constexpr std::size_t block_header_bits =
        shared_exponent_bits +
        characterization_bits;

    constexpr std::size_t value_bits =
        first_field_bits +
        offset_bits +
        stored_mantissa_bits;

    constexpr std::uint8_t first_half_positive_bit = 0;
    constexpr std::uint8_t second_half_positive_bit = 1;

    const std::size_t block_base =
        tensor.block_offsets_[block_index];

    const std::uint8_t exponent_field =
        static_cast<std::uint8_t>(
            tensor.bits_.readBits(
                block_base,
                shared_exponent_bits));

    const int exponent =
        decodeSharedExponent(exponent_field);

    const std::uint8_t characterization =
        static_cast<std::uint8_t>(
            tensor.bits_.readBits(
                block_base + shared_exponent_bits,
                characterization_bits));

    const bool is_first_half =
        value_index < half_block_size;

    const bool positive_half =
        is_first_half ?
            (characterization &
             (std::uint8_t{1} <<
              first_half_positive_bit)) != 0 :
            (characterization &
             (std::uint8_t{1} <<
              second_half_positive_bit)) != 0;

    const std::size_t value_base =
        block_base +
        block_header_bits +
        value_index * value_bits;

    const std::uint8_t first_field =
        static_cast<std::uint8_t>(
            tensor.bits_.readBits(
                value_base,
                first_field_bits));

    const std::uint8_t offset =
        static_cast<std::uint8_t>(
            tensor.bits_.readBits(
                value_base + first_field_bits,
                offset_bits));

    std::uint8_t mantissa =
        static_cast<std::uint8_t>(
            tensor.bits_.readBits(
                value_base +
                    first_field_bits +
                    offset_bits,
                stored_mantissa_bits));

    AFPValue result;

    result.exponent =
        static_cast<std::int8_t>(exponent);

    result.offset = offset;

    if (positive_half)
    {
        /*
         * Positive field:
         *
         * The first field bit is reclaimed as
         * the sixth mantissa bit.
         */
        mantissa |= static_cast<std::uint8_t>(
            first_field << stored_mantissa_bits);

        result.negative = false;

        const int value_mantissa_bits = 6;

        if (result.offset < maximum_offset)
        {
            result.mantissa =
                (std::uint64_t{1} <<
                 value_mantissa_bits) +
                mantissa;
        }
        else
        {
            result.mantissa = mantissa;
        }
    }
    else
    {
        /*
         * Normal field:
         *
         * The first field bit is the sign.
         */
        result.negative =
            first_field != 0;

        const int value_mantissa_bits = 5;

        if (result.offset < maximum_offset)
        {
            result.mantissa =
                (std::uint64_t{1} <<
                 value_mantissa_bits) +
                mantissa;
        }
        else
        {
            result.mantissa = mantissa;
        }
    }

    if (isZeroAFPValue(result))
        result.negative = false;

    return result;
}

void AFPArithmetic::writeAFPValue(
    BitStream &bits,
    const AFPValue &value,
    bool positive_field)
{
    constexpr std::size_t stored_mantissa_bits = 5;

    /*
     * The representation width belongs to the field,
     * not to AFPValue itself.
     */
    const int value_mantissa_bits =
        positive_field ? 6 : 5;

    std::uint64_t mantissa = 0;

    if (value.offset < maximum_offset)
    {
        const std::uint64_t implicit =
            std::uint64_t{1} <<
            value_mantissa_bits;

        if (value.mantissa >= implicit)
        {
            mantissa =
                value.mantissa -
                implicit;
        }
    }
    else
    {
        mantissa = value.mantissa;
    }

    if (positive_field)
    {
        const std::uint64_t extra_bit =
            mantissa >> stored_mantissa_bits;

        const std::uint64_t stored_mantissa =
            mantissa &
            maskBits(stored_mantissa_bits);

        bits.writeBits(extra_bit, 1);

        bits.writeBits(
            static_cast<std::uint64_t>(
                value.offset),
            3);

        bits.writeBits(
            stored_mantissa,
            stored_mantissa_bits);
    }
    else
    {
        bits.writeBits(
            value.negative ? 1 : 0,
            1);

        bits.writeBits(
            static_cast<std::uint64_t>(
                value.offset),
            3);

        bits.writeBits(
            mantissa &
                maskBits(stored_mantissa_bits),
            stored_mantissa_bits);
    }
}

// ============================================================================
// AFP-Normal Arithmetic Operations
// ============================================================================

AFPArithmetic::AFPProduct AFPArithmetic::normalizeToProduct(
    const AFPValue &value)
{
    AFPProduct result;

    if (isZeroAFPValue(value))
        return result;

    result.negative =
        value.negative;

    result.significand =
        value.mantissa;

    const int mantissa_bits =
        mantissaBitsForValue(value);

    result.scale_exponent =
        static_cast<int>(value.exponent) -
        static_cast<int>(value.offset) -
        mantissa_bits;

    result.zero = false;

    return result;
}

AFPArithmetic::AFPProduct AFPArithmetic::multiplyAFPValues(
    const AFPValue &a,
    const AFPValue &b)
{
    AFPProduct result;

    if (isZeroAFPValue(a) ||
        isZeroAFPValue(b))
    {
        return result;
    }

    const AFPProduct norm_a =
        normalizeToProduct(a);

    const AFPProduct norm_b =
        normalizeToProduct(b);

    result.negative =
        norm_a.negative ^
        norm_b.negative;

    result.significand =
        norm_a.significand *
        norm_b.significand;

    result.scale_exponent =
        norm_a.scale_exponent +
        norm_b.scale_exponent;

    result.zero =
        result.significand == 0;

    if (result.zero)
        result.negative = false;

    return result;
}

AFPArithmetic::AFPProduct AFPArithmetic::divideAFPValues(
    const AFPValue &a,
    const AFPValue &b)
{
    AFPProduct result;

    if (isZeroAFPValue(a))
    {
        result.zero = true;
        return result;
    }

    if (isZeroAFPValue(b))
    {
        throw std::invalid_argument(
            "AFP division by zero");
    }

    const AFPProduct numerator =
        normalizeToProduct(a);

    const AFPProduct denominator =
        normalizeToProduct(b);

    if (numerator.zero)
    {
        result.zero = true;
        return result;
    }

    if (denominator.zero ||
        denominator.significand == 0)
    {
        throw std::invalid_argument(
            "AFP division by zero");
    }

    result.negative =
        numerator.negative ^
        denominator.negative;

    constexpr int division_precision_bits = 24;

    const std::uint64_t numerator_significand =
        numerator.significand;

    const std::uint64_t denominator_significand =
        denominator.significand;

    if (numerator_significand >
        (std::numeric_limits<std::uint64_t>::max() >>
         division_precision_bits))
    {
        throw std::overflow_error(
            "AFP division numerator overflow");
    }

    const std::uint64_t scaled_numerator =
        numerator_significand
        << division_precision_bits;

    const std::uint64_t quotient =
        scaled_numerator /
        denominator_significand;

    const std::uint64_t remainder =
        scaled_numerator %
        denominator_significand;

    std::uint64_t rounded_quotient =
        quotient;

    if (remainder >=
        denominator_significand - remainder)
    {
        if (rounded_quotient ==
            std::numeric_limits<std::uint64_t>::max())
        {
            throw std::overflow_error(
                "AFP division quotient overflow");
        }

        ++rounded_quotient;
    }

    if (rounded_quotient == 0)
    {
        result.zero = true;
        result.negative = false;
        result.significand = 0;
        result.scale_exponent = 0;

        return result;
    }

    result.zero = false;

    result.significand =
        rounded_quotient;

    result.scale_exponent =
        numerator.scale_exponent -
        denominator.scale_exponent -
        division_precision_bits;

    return result;
}

AFPArithmetic::AFPAccumulator AFPArithmetic::addProducts(
    const AFPAccumulator &accumulator,
    const AFPProduct &product)
{
    if (product.zero ||
        product.significand == 0)
    {
        return accumulator;
    }

    AFPAccumulator result =
        accumulator;

    if (result.zero ||
        result.significand == 0)
    {
        result.zero = false;

        result.negative =
            product.negative;

        result.significand =
            static_cast<std::int64_t>(
                product.significand);

        result.exponent =
            product.scale_exponent;

        normalizeAccumulator(result);

        return result;
    }

    const int common_exponent =
        std::min(
            result.exponent,
            product.scale_exponent);

    const int accumulator_shift =
        result.exponent -
        common_exponent;

    const int product_shift =
        product.scale_exponent -
        common_exponent;

    std::int64_t accumulator_value =
        result.significand;

    std::int64_t product_value =
        static_cast<std::int64_t>(
            product.significand);

    if (accumulator_shift > 0)
    {
        if (accumulator_shift >= 63)
        {
            accumulator_value = 0;
        }
        else
        {
            accumulator_value <<=
                accumulator_shift;
        }
    }

    if (product_shift > 0)
    {
        if (product_shift >= 63)
        {
            product_value = 0;
        }
        else
        {
            product_value <<=
                product_shift;
        }
    }

    if (result.negative)
    {
        accumulator_value =
            -accumulator_value;
    }

    if (product.negative)
    {
        product_value =
            -product_value;
    }

    const std::int64_t sum =
        accumulator_value +
        product_value;

    if (sum == 0)
    {
        AFPAccumulator zero;

        zero.zero = true;
        zero.negative = false;
        zero.significand = 0;
        zero.exponent = 0;

        return zero;
    }

    AFPAccumulator output;

    output.zero = false;

    output.negative =
        sum < 0;

    if (sum < 0)
    {
        output.significand =
            -sum;
    }
    else
    {
        output.significand =
            sum;
    }

    output.exponent =
        common_exponent;

    normalizeAccumulator(output);

    return output;
}

AFPArithmetic::AFPAccumulator AFPArithmetic::addAFPValues(
    const AFPValue &a,
    const AFPValue &b)
{
    const AFPProduct prod_a =
        normalizeToProduct(a);

    const AFPProduct prod_b =
        normalizeToProduct(b);

    AFPAccumulator result;

    result =
        addProducts(
            result,
            prod_a);

    result =
        addProducts(
            result,
            prod_b);

    return result;
}

AFPArithmetic::AFPAccumulator AFPArithmetic::subtractAFPValues(
    const AFPValue &a,
    const AFPValue &b)
{
    AFPValue neg_b = b;

    if (!isZeroAFPValue(neg_b))
        neg_b.negative =
            !neg_b.negative;

    return addAFPValues(
        a,
        neg_b);
}

// ============================================================================
// AFP Value Conversion
// ============================================================================

AFPArithmetic::AFPValue AFPArithmetic::accumulatorToAFPValue(
    const AFPAccumulator &acc,
    int shared_exponent,
    bool positive_field)
{
    AFPValue result;

    result.exponent =
        static_cast<std::int8_t>(
            shared_exponent);

    result.negative =
        acc.negative;

    const int mantissa_bits =
        positive_field ? 6 : 5;

    if (acc.zero ||
        acc.significand == 0)
    {
        result.negative = false;
        result.offset =
            maximum_offset;
        result.mantissa = 0;

        return result;
    }

    const int highest_bit =
        integerLog2(
            static_cast<std::uint64_t>(
                acc.significand));

    const int value_exponent =
        acc.exponent +
        highest_bit;

    int offset =
        shared_exponent -
        value_exponent;

    if (offset < 0)
        offset = 0;

    if (offset > maximum_offset)
        offset = maximum_offset;

    result.offset =
        static_cast<std::uint8_t>(
            offset);

    const int target_exponent =
        shared_exponent -
        offset -
        mantissa_bits;

    const int shift =
        target_exponent -
        acc.exponent;

    std::uint64_t scaled;

    if (shift >= 0)
    {
        if (shift >= 64)
            scaled = 0;
        else
            scaled =
                shiftRightRounded(
                    static_cast<std::uint64_t>(
                        acc.significand),
                    shift);
    }
    else
    {
        const int left_shift =
            -shift;

        if (left_shift >= 63)
        {
            scaled =
                std::numeric_limits<
                    std::uint64_t>::max();
        }
        else
        {
            scaled =
                static_cast<std::uint64_t>(
                    acc.significand)
                << left_shift;
        }
    }

    if (offset < maximum_offset)
    {
        const std::uint64_t implicit =
            std::uint64_t{1} <<
            mantissa_bits;

        const std::uint64_t maximum =
            (implicit << 1) - 1;

        if (scaled < implicit)
            scaled = implicit;

        if (scaled > maximum)
            scaled = maximum;

        result.mantissa =
            scaled;
    }
    else
    {
        const std::uint64_t maximum =
            maskBits(mantissa_bits);

        if (scaled > maximum)
            scaled = maximum;

        result.mantissa =
            scaled;
    }

    if (isZeroAFPValue(result))
        result.negative = false;

    return result;
}

AFPArithmetic::AFPValue AFPArithmetic::productToAFPValue(
    const AFPProduct &prod,
    int shared_exponent,
    bool positive_field)
{
    AFPAccumulator acc;

    acc.negative =
        prod.negative;

    acc.significand =
        static_cast<std::int64_t>(
            prod.significand);

    acc.exponent =
        prod.scale_exponent;

    acc.zero =
        prod.zero;

    return accumulatorToAFPValue(
        acc,
        shared_exponent,
        positive_field);
}

// ============================================================================
// AFP Activation and Comparison Operations
// ============================================================================

bool AFPArithmetic::isAFPValueZero(
    const AFPValue &value)
{
    return isZeroAFPValue(value);
}

bool AFPArithmetic::isAFPValueNegative(
    const AFPValue &value)
{
    return value.negative &&
           !isZeroAFPValue(value);
}

int AFPArithmetic::compareAFPValues(
    const AFPValue &a,
    const AFPValue &b)
{
    const bool a_zero =
        isZeroAFPValue(a);

    const bool b_zero =
        isZeroAFPValue(b);

    if (a_zero && b_zero)
        return 0;

    if (a_zero)
        return b.negative ? 1 : -1;

    if (b_zero)
        return a.negative ? -1 : 1;

    if (a.negative != b.negative)
        return a.negative ? -1 : 1;

    const AFPProduct pa =
        normalizeToProduct(a);

    const AFPProduct pb =
        normalizeToProduct(b);

    const int a_top =
        pa.scale_exponent +
        integerLog2(pa.significand);

    const int b_top =
        pb.scale_exponent +
        integerLog2(pb.significand);

    int magnitude_comparison = 0;

    if (a_top < b_top)
    {
        magnitude_comparison = -1;
    }
    else if (a_top > b_top)
    {
        magnitude_comparison = 1;
    }
    else if (pa.scale_exponent ==
             pb.scale_exponent)
    {
        if (pa.significand <
            pb.significand)
        {
            magnitude_comparison = -1;
        }
        else if (pa.significand >
                 pb.significand)
        {
            magnitude_comparison = 1;
        }
    }
    else
    {
        const int shift =
            pa.scale_exponent -
            pb.scale_exponent;

        if (shift > 0)
        {
            if (shift >= 64)
            {
                magnitude_comparison = 1;
            }
            else
            {
                const std::uint64_t lhs =
                    pa.significand << shift;

                magnitude_comparison =
                    lhs < pb.significand ? -1 :
                    lhs > pb.significand ? 1 : 0;
            }
        }
        else
        {
            const int reverse_shift =
                -shift;

            if (reverse_shift >= 64)
            {
                magnitude_comparison = -1;
            }
            else
            {
                const std::uint64_t rhs =
                    pb.significand <<
                    reverse_shift;

                magnitude_comparison =
                    pa.significand < rhs ? -1 :
                    pa.significand > rhs ? 1 : 0;
            }
        }
    }

    if (!a.negative)
        return magnitude_comparison;

    return -magnitude_comparison;
}

AFPArithmetic::AFPValue AFPArithmetic::applyReLU(
    const AFPValue &value)
{
    if (isAFPValueNegative(value))
        return zeroValue();

    return value;
}

// ============================================================================
// Block-Level Operations
// ============================================================================

int AFPArithmetic::computeSharedExponent(
    const std::vector<AFPValue> &values,
    std::size_t block_start)
{
    int shared_exponent = -126;
    bool found_nonzero = false;

    for (std::size_t i = 0;
         i < block_size &&
         block_start + i < values.size();
         ++i)
    {
        const AFPValue &value =
            values[block_start + i];

        if (isZeroAFPValue(value))
            continue;

        const int effective_exponent =
            static_cast<int>(value.exponent) -
            static_cast<int>(value.offset);

        if (!found_nonzero ||
            effective_exponent >
                shared_exponent)
        {
            shared_exponent =
                effective_exponent;

            found_nonzero = true;
        }
    }

    return shared_exponent;
}

bool AFPArithmetic::computeHalfPositive(
    const std::vector<AFPValue> &values,
    std::size_t block_start,
    bool first_half)
{
    const std::size_t start =
        first_half ? 0 : half_block_size;

    const std::size_t end =
        first_half ? half_block_size : block_size;

    for (std::size_t i = start;
         i < end &&
         block_start + i < values.size();
         ++i)
    {
        const AFPValue &value =
            values[block_start + i];

        if (!isZeroAFPValue(value) &&
            value.negative)
        {
            return false;
        }
    }

    return true;
}

std::uint8_t AFPArithmetic::buildCharacterization(
    bool first_half_positive,
    bool second_half_positive)
{
    constexpr std::uint8_t first_half_positive_bit = 0;
    constexpr std::uint8_t second_half_positive_bit = 1;

    std::uint8_t characterization = 0;

    if (first_half_positive)
    {
        characterization |=
            (std::uint8_t{1} <<
             first_half_positive_bit);
    }

    if (second_half_positive)
    {
        characterization |=
            (std::uint8_t{1} <<
             second_half_positive_bit);
    }

    return characterization;
}

// ============================================================================
// Tensor Construction
// ============================================================================

AFPEncodedTensor AFPArithmetic::buildTensorFromAFPValues(
    const std::vector<AFPValue> &values,
    const AFPConfig &config)
{
    AFPEncodedTensor result;

    result.config_ = config;
    result.value_count_ = values.size();

    if (values.empty())
        return result;

    for (std::size_t block_start = 0;
         block_start < values.size();
         block_start += block_size)
    {
        result.block_offsets_.push_back(
            result.bits_.bitSize());

        const int shared_exponent =
            computeSharedExponent(
                values,
                block_start);

        const bool first_half_positive =
            computeHalfPositive(
                values,
                block_start,
                true);

        const bool second_half_positive =
            computeHalfPositive(
                values,
                block_start,
                false);

        const std::uint8_t characterization =
            buildCharacterization(
                first_half_positive,
                second_half_positive);

        result.bits_.writeBits(
            encodeSharedExponent(
                shared_exponent),
            8);

        result.bits_.writeBits(
            characterization,
            8);

        for (std::size_t i = 0;
             i < block_size;
             ++i)
        {
            const bool positive_field =
                i < half_block_size ?
                    first_half_positive :
                    second_half_positive;

            AFPValue value;

            if (block_start + i < values.size())
            {
                const AFPProduct normalized =
                    normalizeToProduct(
                        values[block_start + i]);

                AFPAccumulator acc;

                acc.negative =
                    normalized.negative;

                acc.significand =
                    static_cast<std::int64_t>(
                        normalized.significand);

                acc.exponent =
                    normalized.scale_exponent;

                acc.zero =
                    normalized.zero;

                value =
                    accumulatorToAFPValue(
                        acc,
                        shared_exponent,
                        positive_field);
            }
            else
            {
                value.exponent =
                    static_cast<std::int8_t>(
                        shared_exponent);

                value.negative = false;
                value.offset =
                    maximum_offset;
                value.mantissa = 0;
            }

            writeAFPValue(
                result.bits_,
                value,
                positive_field);
        }
    }

    return result;
}

// ============================================================================
// Public Interface - Element-wise Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::add(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    validateCompatible(a, b);

    std::vector<AFPValue> output;
    output.reserve(a.size());

    for (std::size_t index = 0;
         index < a.size();
         ++index)
    {
        const std::size_t block =
            index / block_size;

        const std::size_t position =
            index % block_size;

        const AFPValue a_value =
            readAFPValue(
                a,
                block,
                position);

        const AFPValue b_value =
            readAFPValue(
                b,
                block,
                position);

        const AFPAccumulator acc =
            addAFPValues(
                a_value,
                b_value);

        int temp_exponent =
            acc.zero ?
                -126 :
                acc.exponent +
                integerLog2(
                    static_cast<std::uint64_t>(
                        acc.significand));

        output.push_back(
            accumulatorToAFPValue(
                acc,
                temp_exponent,
                false));
    }

    return buildTensorFromAFPValues(
        output,
        a.config_);
}

AFPEncodedTensor AFPArithmetic::subtract(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    validateCompatible(a, b);

    std::vector<AFPValue> output;
    output.reserve(a.size());

    for (std::size_t index = 0;
         index < a.size();
         ++index)
    {
        const std::size_t block =
            index / block_size;

        const std::size_t position =
            index % block_size;

        const AFPValue a_value =
            readAFPValue(
                a,
                block,
                position);

        const AFPValue b_value =
            readAFPValue(
                b,
                block,
                position);

        const AFPAccumulator acc =
            subtractAFPValues(
                a_value,
                b_value);

        int temp_exponent =
            acc.zero ?
                -126 :
                acc.exponent +
                integerLog2(
                    static_cast<std::uint64_t>(
                        acc.significand));

        output.push_back(
            accumulatorToAFPValue(
                acc,
                temp_exponent,
                false));
    }

    return buildTensorFromAFPValues(
        output,
        a.config_);
}

AFPEncodedTensor AFPArithmetic::multiply(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    validateCompatible(a, b);

    std::vector<AFPValue> output;
    output.reserve(a.size());

    for (std::size_t index = 0;
         index < a.size();
         ++index)
    {
        const std::size_t block =
            index / block_size;

        const std::size_t position =
            index % block_size;

        const AFPValue a_value =
            readAFPValue(
                a,
                block,
                position);

        const AFPValue b_value =
            readAFPValue(
                b,
                block,
                position);

        const AFPProduct product =
            multiplyAFPValues(
                a_value,
                b_value);

        AFPAccumulator acc;

        acc.negative =
            product.negative;

        acc.significand =
            static_cast<std::int64_t>(
                product.significand);

        acc.exponent =
            product.scale_exponent;

        acc.zero =
            product.zero;

        int temp_exponent =
            acc.zero ?
                -126 :
                acc.exponent +
                integerLog2(
                    static_cast<std::uint64_t>(
                        acc.significand));

        output.push_back(
            accumulatorToAFPValue(
                acc,
                temp_exponent,
                false));
    }

    return buildTensorFromAFPValues(
        output,
        a.config_);
}

AFPEncodedTensor AFPArithmetic::divide(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    validateCompatible(a, b);

    std::vector<AFPValue> output;
    output.reserve(a.size());

    for (std::size_t index = 0;
         index < a.size();
         ++index)
    {
        const std::size_t block =
            index / block_size;

        const std::size_t position =
            index % block_size;

        const AFPValue a_value =
            readAFPValue(
                a,
                block,
                position);

        const AFPValue b_value =
            readAFPValue(
                b,
                block,
                position);

        const AFPProduct quotient =
            divideAFPValues(
                a_value,
                b_value);

        AFPAccumulator acc;

        acc.negative =
            quotient.negative;

        acc.significand =
            static_cast<std::int64_t>(
                quotient.significand);

        acc.exponent =
            quotient.scale_exponent;

        acc.zero =
            quotient.zero;

        int temp_exponent =
            acc.zero ?
                -126 :
                acc.exponent +
                integerLog2(
                    static_cast<std::uint64_t>(
                        acc.significand));

        output.push_back(
            accumulatorToAFPValue(
                acc,
                temp_exponent,
                false));
    }

    return buildTensorFromAFPValues(
        output,
        a.config_);
}

AFPEncodedTensor AFPArithmetic::negate(
    const AFPEncodedTensor &input)
{
    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        if (!isZeroAFPValue(value))
            value.negative =
                !value.negative;

        output.push_back(value);
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::absolute(
    const AFPEncodedTensor &input)
{
    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        value.negative = false;

        output.push_back(value);
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::minimum(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    validateCompatible(a, b);

    std::vector<AFPValue> output;
    output.reserve(a.size());

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        const AFPValue av =
            readAFPValue(
                a,
                i / block_size,
                i % block_size);

        const AFPValue bv =
            readAFPValue(
                b,
                i / block_size,
                i % block_size);

        output.push_back(
            compareAFPValues(av, bv) <= 0 ?
                av :
                bv);
    }

    return buildTensorFromAFPValues(
        output,
        a.config_);
}

AFPEncodedTensor AFPArithmetic::maximum(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    validateCompatible(a, b);

    std::vector<AFPValue> output;
    output.reserve(a.size());

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        const AFPValue av =
            readAFPValue(
                a,
                i / block_size,
                i % block_size);

        const AFPValue bv =
            readAFPValue(
                b,
                i / block_size,
                i % block_size);

        output.push_back(
            compareAFPValues(av, bv) >= 0 ?
                av :
                bv);
    }

    return buildTensorFromAFPValues(
        output,
        a.config_);
}

AFPEncodedTensor AFPArithmetic::clamp(
    const AFPEncodedTensor &input,
    const AFPEncodedTensor &min_value,
    const AFPEncodedTensor &max_value)
{
    if (min_value.size() != 1)
        throw std::invalid_argument(
            "AFP clamp: min_value must contain one value");

    if (max_value.size() != 1)
        throw std::invalid_argument(
            "AFP clamp: max_value must contain one value");

    if (input.config_.block_size !=
            min_value.config_.block_size ||
        input.config_.block_size !=
            max_value.config_.block_size)
    {
        throw std::invalid_argument(
            "AFP clamp: block size mismatch");
    }

    const AFPValue min_val =
        readAFPValue(
            min_value,
            0,
            0);

    const AFPValue max_val =
        readAFPValue(
            max_value,
            0,
            0);

    if (compareAFPValues(
            min_val,
            max_val) > 0)
    {
        throw std::invalid_argument(
            "AFP clamp: min_value greater than max_value");
    }

    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        if (compareAFPValues(
                value,
                min_val) < 0)
        {
            output.push_back(min_val);
        }
        else if (compareAFPValues(
                     value,
                     max_val) > 0)
        {
            output.push_back(max_val);
        }
        else
        {
            output.push_back(value);
        }
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::leakyRelu(
    const AFPEncodedTensor &input,
    const AFPEncodedTensor &alpha)
{
    if (alpha.size() != 1)
        throw std::invalid_argument(
            "AFP leakyRelu: alpha must contain one value");

    const AFPValue alpha_value =
        readAFPValue(
            alpha,
            0,
            0);

    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        if (isZeroAFPValue(value) ||
            !value.negative)
        {
            output.push_back(value);
            continue;
        }

        output.push_back(
            multiplyValues(
                value,
                alpha_value));
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::silu(
    const AFPEncodedTensor &input)
{
    AFPEncodedTensor sigmoid_result =
        sigmoid(input);

    return multiply(
        input,
        sigmoid_result);
}

// ============================================================================
// Public Interface - Dot Product
// ============================================================================

AFPEncodedTensor AFPArithmetic::dotProduct(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    validateCompatible(a, b);

    if (a.size() == 0)
    {
        std::vector<AFPValue> zero_values(1);

        zero_values[0] =
            zeroValue();

        return buildTensorFromAFPValues(
            zero_values,
            a.config_);
    }

    AFPAccumulator accumulator;

    for (std::size_t index = 0;
         index < a.size();
         ++index)
    {
        const std::size_t block =
            index / block_size;

        const std::size_t position =
            index % block_size;

        const AFPValue a_value =
            readAFPValue(
                a,
                block,
                position);

        const AFPValue b_value =
            readAFPValue(
                b,
                block,
                position);

        const AFPProduct product =
            multiplyAFPValues(
                a_value,
                b_value);

        accumulator =
            addProducts(
                accumulator,
                product);
    }

    std::vector<AFPValue> result_values;

    if (accumulator.zero)
    {
        result_values.push_back(
            zeroValue());
    }
    else
    {
        const int exponent =
            accumulator.exponent +
            integerLog2(
                static_cast<std::uint64_t>(
                    accumulator.significand));

        result_values.push_back(
            accumulatorToAFPValue(
                accumulator,
                exponent,
                false));
    }

    return buildTensorFromAFPValues(
        result_values,
        a.config_);
}

// ============================================================================
// Public Interface - Matrix Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::matrixVectorMultiply(
    const AFPEncodedTensor &weights,
    const AFPEncodedTensor &input,
    std::size_t rows,
    std::size_t columns)
{
    if (weights.size() !=
        rows * columns)
    {
        throw std::invalid_argument(
            "Weight tensor size doesn't match dimensions");
    }

    if (input.size() != columns)
    {
        throw std::invalid_argument(
            "Input tensor size doesn't match columns");
    }

    std::vector<AFPValue> output;
    output.reserve(rows);

    for (std::size_t row = 0;
         row < rows;
         ++row)
    {
        AFPAccumulator row_accumulator;

        for (std::size_t col = 0;
             col < columns;
             ++col)
        {
            const std::size_t weight_idx =
                row * columns + col;

            const std::size_t weight_block =
                weight_idx / block_size;

            const std::size_t weight_pos =
                weight_idx % block_size;

            const AFPValue weight_value =
                readAFPValue(
                    weights,
                    weight_block,
                    weight_pos);

            const std::size_t input_block =
                col / block_size;

            const std::size_t input_pos =
                col % block_size;

            const AFPValue input_value =
                readAFPValue(
                    input,
                    input_block,
                    input_pos);

            const AFPProduct product =
                multiplyAFPValues(
                    weight_value,
                    input_value);

            row_accumulator =
                addProducts(
                    row_accumulator,
                    product);
        }

        if (row_accumulator.zero)
        {
            output.push_back(
                zeroValue());
        }
        else
        {
            const int exponent =
                row_accumulator.exponent +
                integerLog2(
                    static_cast<std::uint64_t>(
                        row_accumulator.significand));

            output.push_back(
                accumulatorToAFPValue(
                    row_accumulator,
                    exponent,
                    false));
        }
    }

    return buildTensorFromAFPValues(
        output,
        weights.config_);
}

AFPEncodedTensor AFPArithmetic::matrixMultiply(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b,
    std::size_t rows_a,
    std::size_t cols_a,
    std::size_t cols_b)
{
    if (a.size() !=
        rows_a * cols_a)
    {
        throw std::invalid_argument(
            "Matrix A size doesn't match dimensions");
    }

    if (b.size() !=
        cols_a * cols_b)
    {
        throw std::invalid_argument(
            "Matrix B size doesn't match dimensions");
    }

    std::vector<AFPValue> output;

    output.reserve(
        rows_a * cols_b);

    for (std::size_t i = 0;
         i < rows_a;
         ++i)
    {
        for (std::size_t j = 0;
             j < cols_b;
             ++j)
        {
            AFPAccumulator element_accumulator;

            for (std::size_t k = 0;
                 k < cols_a;
                 ++k)
            {
                const std::size_t a_idx =
                    i * cols_a + k;

                const std::size_t b_idx =
                    k * cols_b + j;

                const std::size_t a_block =
                    a_idx / block_size;

                const std::size_t a_pos =
                    a_idx % block_size;

                const AFPValue a_value =
                    readAFPValue(
                        a,
                        a_block,
                        a_pos);

                const std::size_t b_block =
                    b_idx / block_size;

                const std::size_t b_pos =
                    b_idx % block_size;

                const AFPValue b_value =
                    readAFPValue(
                        b,
                        b_block,
                        b_pos);

                const AFPProduct product =
                    multiplyAFPValues(
                        a_value,
                        b_value);

                element_accumulator =
                    addProducts(
                        element_accumulator,
                        product);
            }

            if (element_accumulator.zero)
            {
                output.push_back(
                    zeroValue());
            }
            else
            {
                const int exponent =
                    element_accumulator.exponent +
                    integerLog2(
                        static_cast<std::uint64_t>(
                            element_accumulator.significand));

                output.push_back(
                    accumulatorToAFPValue(
                        element_accumulator,
                        exponent,
                        false));
            }
        }
    }

    return buildTensorFromAFPValues(
        output,
        a.config_);
}

// ============================================================================
// Public Interface - Activation Functions
// ============================================================================

AFPEncodedTensor AFPArithmetic::relu(
    const AFPEncodedTensor &input)
{
    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t index = 0;
         index < input.size();
         ++index)
    {
        const std::size_t block =
            index / block_size;

        const std::size_t position =
            index % block_size;

        AFPValue value =
            readAFPValue(
                input,
                block,
                position);

        output.push_back(
            applyReLU(value));
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::sigmoid(
    const AFPEncodedTensor &input)
{
    std::vector<AFPValue> output;
    output.reserve(input.size());

    const auto makeConstant =
        [](int exponent,
           std::uint64_t mantissa,
           bool negative = false)
    {
        AFPValue result;

        result.negative = negative;

        result.exponent =
            static_cast<std::int8_t>(
                exponent);

        result.offset = 0;
        result.mantissa = mantissa;

        return result;
    };

    const AFPProduct half =
        normalizeToProduct(
            makeConstant(-1, 32));

    const AFPProduct quarter =
        normalizeToProduct(
            makeConstant(-2, 32));

    const AFPProduct inverse48 =
        normalizeToProduct(
            makeConstant(-6, 43));

    for (std::size_t index = 0;
         index < input.size();
         ++index)
    {
        const std::size_t block =
            index / block_size;

        const std::size_t position =
            index % block_size;

        AFPValue value =
            readAFPValue(
                input,
                block,
                position);

        const AFPProduct prod =
            normalizeToProduct(value);

        if (isZeroAFPValue(value))
        {
            output.push_back(
                makeConstant(-1, 32));

            continue;
        }

        if (prod.scale_exponent >= 2)
        {
            output.push_back(
                value.negative ?
                    makeConstant(-1, 0) :
                    makeConstant(0, 32));

            if (value.negative)
                output.back().offset =
                    maximum_offset;

            continue;
        }

        const AFPProduct x2 =
            multiplyAFPValues(
                value,
                value);

        AFPProduct x3 = x2;

        x3.significand *=
            prod.significand;

        x3.scale_exponent +=
            prod.scale_exponent;

        x3.negative =
            prod.negative;

        x3.significand *=
            inverse48.significand;

        x3.scale_exponent +=
            inverse48.scale_exponent;

        AFPAccumulator acc;

        acc =
            addProducts(
                acc,
                half);

        AFPProduct linear =
            prod;

        linear.significand *=
            quarter.significand;

        linear.scale_exponent +=
            quarter.scale_exponent;

        acc =
            addProducts(
                acc,
                linear);

        x3.negative =
            !x3.negative;

        acc =
            addProducts(
                acc,
                x3);

        const int exponent =
            acc.exponent +
            integerLog2(
                static_cast<std::uint64_t>(
                    acc.significand));

        output.push_back(
            accumulatorToAFPValue(
                acc,
                exponent,
                false));
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::tanh(
    const AFPEncodedTensor &input)
{
    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t index = 0;
         index < input.size();
         ++index)
    {
        const std::size_t block =
            index / block_size;

        const std::size_t position =
            index % block_size;

        AFPValue value =
            readAFPValue(
                input,
                block,
                position);

        if (isZeroAFPValue(value))
        {
            output.push_back(value);
            continue;
        }

        const AFPProduct product =
            normalizeToProduct(value);

        const int highest =
            product.scale_exponent +
            integerLog2(
                product.significand);

        bool large = false;

        if (highest > 1)
        {
            large = true;
        }
        else if (highest == 1)
        {
            const int shift =
                1 -
                product.scale_exponent;

            if (shift >= 0 &&
                shift < 64)
            {
                large =
                    product.significand >=
                    (std::uint64_t{3} <<
                     shift);
            }
        }

        if (large)
        {
            AFPValue result;

            result.negative =
                value.negative;

            result.exponent = 0;
            result.offset = 0;

            result.mantissa =
                std::uint64_t{1} << 5;

            output.push_back(result);

            continue;
        }

        const auto fromProduct =
            [](const AFPProduct &p)
        {
            AFPAccumulator accumulator;

            accumulator.zero =
                p.zero;

            accumulator.negative =
                p.negative;

            accumulator.significand =
                static_cast<std::int64_t>(
                    p.significand);

            accumulator.exponent =
                p.scale_exponent;

            return accumulator;
        };

        const auto toProduct =
            [](const AFPAccumulator &a)
        {
            AFPProduct product;

            product.zero =
                a.zero;

            product.negative =
                a.negative;

            product.significand =
                static_cast<std::uint64_t>(
                    a.significand);

            product.scale_exponent =
                a.exponent;

            return product;
        };

        const auto constant =
            [](std::uint64_t constant_value)
        {
            AFPProduct product;

            product.zero = false;
            product.negative = false;

            product.significand =
                constant_value;

            product.scale_exponent = 0;

            return product;
        };

        const AFPProduct square =
            multiplyAFPValues(
                value,
                value);

        AFPAccumulator numerator_part_accumulator =
            addProducts(
                fromProduct(
                    constant(27)),
                square);

        AFPProduct numerator_part_product =
            toProduct(
                numerator_part_accumulator);

        AFPValue numerator_part_value;

        if (numerator_part_product.zero ||
            numerator_part_product.significand == 0)
        {
            numerator_part_value =
                zeroValue();
        }
        else
        {
            const int numerator_part_exponent =
                numerator_part_product.scale_exponent +
                integerLog2(
                    numerator_part_product.significand);

            numerator_part_value =
                productToAFPValue(
                    numerator_part_product,
                    numerator_part_exponent,
                    false);
        }

        const AFPProduct numerator =
            multiplyAFPValues(
                value,
                numerator_part_value);

        AFPProduct nine_square =
            square;

        nine_square.significand *= 9;

        AFPAccumulator denominator_accumulator =
            addProducts(
                fromProduct(
                    constant(27)),
                nine_square);

        AFPProduct denominator =
            toProduct(
                denominator_accumulator);

        if (denominator.zero ||
            denominator.significand == 0)
        {
            output.push_back(
                zeroValue());

            continue;
        }

        constexpr int precision = 24;

        std::uint64_t numerator_significand =
            numerator.significand;

        int numerator_scale =
            numerator.scale_exponent;

        (void)numerator_scale;

        constexpr int max_shift = 63;

        int available_shift =
            max_shift -
            integerLog2(
                numerator_significand);

        if (available_shift < 0)
            available_shift = 0;

        const int actual_precision =
            precision < available_shift ?
                precision :
                available_shift;

        const std::uint64_t scaled_numerator =
            numerator_significand <<
            actual_precision;

        AFPProduct result;

        result.zero =
            numerator.zero ||
            numerator.significand == 0;

        result.negative =
            numerator.negative ^
            denominator.negative;

        if (result.zero)
        {
            result.significand = 0;
            result.scale_exponent = 0;
        }
        else
        {
            result.significand =
                scaled_numerator /
                denominator.significand;

            result.scale_exponent =
                numerator.scale_exponent -
                denominator.scale_exponent -
                actual_precision;

            if (result.significand == 0)
            {
                result.zero = true;
                result.negative = false;
            }
        }

        if (result.zero)
        {
            output.push_back(
                zeroValue());
        }
        else
        {
            const int result_exponent =
                result.scale_exponent +
                integerLog2(
                    result.significand);

            output.push_back(
                productToAFPValue(
                    result,
                    result_exponent,
                    false));
        }
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::gelu(
    const AFPEncodedTensor &input)
{
    std::vector<AFPValue> output;
    output.reserve(input.size());

    const AFPValue half =
        halfValue();

    AFPValue three_quarters;

    three_quarters.negative = false;
    three_quarters.exponent = -1;
    three_quarters.offset = 0;
    three_quarters.mantissa =
        (std::uint64_t{1} << 5) + 16;

    AFPValue one_sixteenth;

    one_sixteenth.negative = false;
    one_sixteenth.exponent = -4;
    one_sixteenth.offset = 0;
    one_sixteenth.mantissa =
        std::uint64_t{1} << 5;

    const AFPValue one =
        oneValue();

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue x =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        const AFPValue x2 =
            multiplyValues(x, x);

        const AFPValue x3 =
            multiplyValues(x2, x);

        const AFPValue cubic =
            multiplyValues(
                x3,
                one_sixteenth);

        const AFPValue inner =
            addValues(
                x,
                cubic);

        const AFPValue scaled =
            multiplyValues(
                inner,
                three_quarters);

        const AFPEncodedTensor tanh_tensor =
            tanh(
                buildTensorFromAFPValues(
                    {scaled},
                    input.config_));

        const AFPValue t =
            tanh_tensor.size() == 0 ?
                zeroValue() :
                readAFPValue(
                    tanh_tensor,
                    0,
                    0);

        const AFPValue one_plus_t =
            addValues(
                one,
                t);

        const AFPValue half_x =
            multiplyValues(
                x,
                half);

        output.push_back(
            multiplyValues(
                half_x,
                one_plus_t));
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPArithmetic::AFPValue AFPArithmetic::expValue(
    const AFPValue &value)
{
    if (isZeroAFPValue(value))
        return oneValue();

    const AFPValue one =
        oneValue();

    AFPValue half;

    half.negative = false;
    half.exponent = -1;
    half.offset = 0;
    half.mantissa = 32;

    AFPValue one_sixth;

    one_sixth.negative = false;
    one_sixth.exponent = -3;
    one_sixth.offset = 0;
    one_sixth.mantissa = 43;

    AFPValue one_twenty_fourth;

    one_twenty_fourth.negative = false;
    one_twenty_fourth.exponent = -5;
    one_twenty_fourth.offset = 0;
    one_twenty_fourth.mantissa = 43;

    AFPValue one_one_twentieth;

    one_one_twentieth.negative = false;
    one_one_twentieth.exponent = -7;
    one_one_twentieth.offset = 0;
    one_one_twentieth.mantissa = 43;

    AFPValue one_seven_twentieth;

    one_seven_twentieth.negative = false;
    one_seven_twentieth.exponent = -9;
    one_seven_twentieth.offset = 0;
    one_seven_twentieth.mantissa = 45;

    const AFPValue x2 =
        multiplyValues(
            value,
            value);

    const AFPValue x3 =
        multiplyValues(
            x2,
            value);

    const AFPValue x4 =
        multiplyValues(
            x3,
            value);

    const AFPValue x5 =
        multiplyValues(
            x4,
            value);

    const AFPValue x6 =
        multiplyValues(
            x5,
            value);

    const AFPValue term2 =
        multiplyValues(
            x2,
            half);

    const AFPValue term3 =
        multiplyValues(
            x3,
            one_sixth);

    const AFPValue term4 =
        multiplyValues(
            x4,
            one_twenty_fourth);

    const AFPValue term5 =
        multiplyValues(
            x5,
            one_one_twentieth);

    const AFPValue term6 =
        multiplyValues(
            x6,
            one_seven_twentieth);

    AFPValue result =
        addValues(
            one,
            value);

    result =
        addValues(
            result,
            term2);

    result =
        addValues(
            result,
            term3);

    result =
        addValues(
            result,
            term4);

    result =
        addValues(
            result,
            term5);

    result =
        addValues(
            result,
            term6);

    return result;
}

AFPEncodedTensor AFPArithmetic::exp(
    const AFPEncodedTensor &input)
{
    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        output.push_back(
            expValue(value));
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::softmax(
    const AFPEncodedTensor &input)
{
    if (input.size() == 0)
        return input;

    AFPValue maximum_value =
        readAFPValue(
            input,
            0,
            0);

    for (std::size_t i = 1;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        if (compareAFPValues(
                value,
                maximum_value) > 0)
        {
            maximum_value = value;
        }
    }

    std::vector<AFPValue> exponentials;
    exponentials.reserve(input.size());

    AFPAccumulator denominator;

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        const AFPValue shifted =
            subtractValues(
                value,
                maximum_value);

        const AFPValue e =
            expValue(shifted);

        exponentials.push_back(e);

        const AFPProduct product =
            normalizeToProduct(e);

        denominator =
            addProducts(
                denominator,
                product);
    }

    if (denominator.zero)
    {
        return buildTensorFromAFPValues(
            std::vector<AFPValue>(
                input.size(),
                zeroValue()),
            input.config_);
    }

    const int denominator_exponent =
        denominator.exponent +
        integerLog2(
            static_cast<std::uint64_t>(
                denominator.significand));

    const AFPValue denominator_value =
        accumulatorToAFPValue(
            denominator,
            denominator_exponent,
            false);

    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (const AFPValue &e :
         exponentials)
    {
        output.push_back(
            divideValues(
                e,
                denominator_value));
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

// ============================================================================
// AFP Scalar Helpers
// ============================================================================

AFPArithmetic::AFPValue AFPArithmetic::zeroValue()
{
    AFPValue value;

    value.negative = false;
    value.exponent = -126;
    value.offset = maximum_offset;
    value.mantissa = 0;

    return value;
}

AFPArithmetic::AFPValue AFPArithmetic::oneValue()
{
    AFPValue value;

    value.negative = false;
    value.exponent = 0;
    value.offset = 0;
    value.mantissa =
        std::uint64_t{1} << 5;

    return value;
}

AFPArithmetic::AFPValue AFPArithmetic::twoValue()
{
    AFPValue value;

    value.negative = false;
    value.exponent = 1;
    value.offset = 0;
    value.mantissa =
        std::uint64_t{1} << 5;

    return value;
}

AFPArithmetic::AFPValue AFPArithmetic::halfValue()
{
    AFPValue value;

    value.negative = false;
    value.exponent = -1;
    value.offset = 0;
    value.mantissa =
        std::uint64_t{1} << 5;

    return value;
}

AFPArithmetic::AFPValue AFPArithmetic::addValues(
    const AFPValue &a,
    const AFPValue &b)
{
    const AFPAccumulator acc =
        addAFPValues(a, b);

    if (acc.zero)
        return zeroValue();

    const int exponent =
        acc.exponent +
        integerLog2(
            static_cast<std::uint64_t>(
                acc.significand));

    return accumulatorToAFPValue(
        acc,
        exponent,
        false);
}

AFPArithmetic::AFPValue AFPArithmetic::subtractValues(
    const AFPValue &a,
    const AFPValue &b)
{
    const AFPAccumulator acc =
        subtractAFPValues(a, b);

    if (acc.zero)
        return zeroValue();

    const int exponent =
        acc.exponent +
        integerLog2(
            static_cast<std::uint64_t>(
                acc.significand));

    return accumulatorToAFPValue(
        acc,
        exponent,
        false);
}

AFPArithmetic::AFPValue AFPArithmetic::multiplyValues(
    const AFPValue &a,
    const AFPValue &b)
{
    const AFPProduct product =
        multiplyAFPValues(a, b);

    if (product.zero)
        return zeroValue();

    AFPAccumulator acc;

    acc.negative =
        product.negative;

    acc.significand =
        static_cast<std::int64_t>(
            product.significand);

    acc.exponent =
        product.scale_exponent;

    acc.zero = false;

    const int exponent =
        acc.exponent +
        integerLog2(
            static_cast<std::uint64_t>(
                acc.significand));

    return accumulatorToAFPValue(
        acc,
        exponent,
        false);
}

AFPArithmetic::AFPValue AFPArithmetic::divideValues(
    const AFPValue &a,
    const AFPValue &b)
{
    const AFPProduct product =
        divideAFPValues(a, b);

    if (product.zero)
        return zeroValue();

    AFPAccumulator acc;

    acc.negative =
        product.negative;

    acc.significand =
        static_cast<std::int64_t>(
            product.significand);

    acc.exponent =
        product.scale_exponent;

    acc.zero = false;

    const int exponent =
        acc.exponent +
        integerLog2(
            static_cast<std::uint64_t>(
                acc.significand));

    return accumulatorToAFPValue(
        acc,
        exponent,
        false);
}

AFPArithmetic::AFPValue AFPArithmetic::scalePowerOfTwo(
    const AFPValue &value,
    int power)
{
    if (isZeroAFPValue(value))
        return value;

    AFPValue result = value;

    int exponent =
        static_cast<int>(
            result.exponent) +
        power;

    if (exponent < -126)
        exponent = -126;

    if (exponent > 127)
        exponent = 127;

    result.exponent =
        static_cast<std::int8_t>(
            exponent);

    return result;
}

// ============================================================================
// Public Interface - Reduction Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::sum(
    const AFPEncodedTensor &input)
{
    if (input.size() == 0)
    {
        std::vector<AFPValue> zero_values(1);

        zero_values[0] =
            zeroValue();

        return buildTensorFromAFPValues(
            zero_values,
            input.config_);
    }

    AFPAccumulator total;

    for (std::size_t index = 0;
         index < input.size();
         ++index)
    {
        const std::size_t block =
            index / block_size;

        const std::size_t position =
            index % block_size;

        const AFPValue value =
            readAFPValue(
                input,
                block,
                position);

        const AFPProduct product =
            normalizeToProduct(value);

        total =
            addProducts(
                total,
                product);
    }

    std::vector<AFPValue> result_values;

    if (total.zero)
    {
        result_values.push_back(
            zeroValue());
    }
    else
    {
        const int exponent =
            total.exponent +
            integerLog2(
                static_cast<std::uint64_t>(
                    total.significand));

        result_values.push_back(
            accumulatorToAFPValue(
                total,
                exponent,
                false));
    }

    return buildTensorFromAFPValues(
        result_values,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::mean(
    const AFPEncodedTensor &input)
{
    if (input.size() == 0)
    {
        return buildTensorFromAFPValues(
            {zeroValue()},
            input.config_);
    }

    AFPAccumulator accumulator;

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        const AFPProduct product =
            normalizeToProduct(value);

        accumulator =
            addProducts(
                accumulator,
                product);
    }

    if (accumulator.zero)
    {
        return buildTensorFromAFPValues(
            {zeroValue()},
            input.config_);
    }

    const int exponent =
        accumulator.exponent +
        integerLog2(
            static_cast<std::uint64_t>(
                accumulator.significand));

    AFPValue sum_value =
        accumulatorToAFPValue(
            accumulator,
            exponent,
            false);

    AFPValue count;

    count.negative = false;
    count.offset = 0;

    const std::size_t n =
        input.size();

    const int n_exponent =
        integerLog2(
            static_cast<std::uint64_t>(
                n));

    count.exponent =
        static_cast<std::int8_t>(
            n_exponent);

    const std::uint64_t normalized_n =
        static_cast<std::uint64_t>(n)
        << 5;

    const int shift =
        n_exponent;

    count.mantissa =
        shiftRightRounded(
            normalized_n,
            shift);

    if (count.mantissa == 0)
        count.mantissa = 1;

    AFPValue mean_value =
        divideValues(
            sum_value,
            count);

    return buildTensorFromAFPValues(
        {mean_value},
        input.config_);
}

AFPEncodedTensor AFPArithmetic::max(
    const AFPEncodedTensor &input)
{
    if (input.size() == 0)
    {
        std::vector<AFPValue> zero_values(1);

        zero_values[0] =
            zeroValue();

        return buildTensorFromAFPValues(
            zero_values,
            input.config_);
    }

    AFPValue max_value =
        readAFPValue(
            input,
            0,
            0);

    for (std::size_t index = 1;
         index < input.size();
         ++index)
    {
        const std::size_t block =
            index / block_size;

        const std::size_t position =
            index % block_size;

        const AFPValue current =
            readAFPValue(
                input,
                block,
                position);

        const AFPProduct max_prod =
            normalizeToProduct(
                max_value);

        const AFPProduct curr_prod =
            normalizeToProduct(
                current);

        if (isZeroAFPValue(current))
            continue;

        if (isZeroAFPValue(max_value))
        {
            max_value = current;
            continue;
        }

        if (max_prod.scale_exponent >
            curr_prod.scale_exponent)
        {
            if (!max_value.negative)
                continue;
            else if (!current.negative)
            {
                max_value = current;
                continue;
            }
            else
            {
                max_value = current;
                continue;
            }
        }
        else if (curr_prod.scale_exponent >
                 max_prod.scale_exponent)
        {
            if (!current.negative)
            {
                max_value = current;
                continue;
            }
            else if (!max_value.negative)
            {
                continue;
            }
            else
            {
                continue;
            }
        }
        else
        {
            if (max_prod.significand >
                curr_prod.significand)
            {
                if (!max_value.negative)
                    continue;
                else
                {
                    max_value = current;
                    continue;
                }
            }
            else if (curr_prod.significand >
                     max_prod.significand)
            {
                if (!current.negative)
                {
                    max_value = current;
                    continue;
                }
                else
                {
                    continue;
                }
            }
        }
    }

    std::vector<AFPValue> result_values(
        1,
        max_value);

    return buildTensorFromAFPValues(
        result_values,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::sumSquares(
    const AFPEncodedTensor &input)
{
    AFPAccumulator accumulator;

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        const AFPProduct square =
            multiplyAFPValues(
                value,
                value);

        accumulator =
            addProducts(
                accumulator,
                square);
    }

    std::vector<AFPValue> result;

    if (accumulator.zero)
    {
        result.push_back(
            zeroValue());
    }
    else
    {
        const int exponent =
            accumulator.exponent +
            integerLog2(
                static_cast<std::uint64_t>(
                    accumulator.significand));

        result.push_back(
            accumulatorToAFPValue(
                accumulator,
                exponent,
                false));
    }

    return buildTensorFromAFPValues(
        result,
        input.config_);
}

AFPArithmetic::AFPValue AFPArithmetic::reciprocalValue(
    const AFPValue &value)
{
    if (isZeroAFPValue(value))
        throw std::invalid_argument(
            "AFP reciprocal: division by zero");

    const AFPProduct p =
        normalizeToProduct(value);

    const int top =
        p.scale_exponent +
        integerLog2(
            p.significand);

    AFPValue estimate;

    estimate.negative =
        value.negative;

    estimate.exponent =
        static_cast<std::int8_t>(
            -top);

    estimate.offset = 0;

    estimate.mantissa =
        std::uint64_t{1} << 5;

    const AFPValue two =
        twoValue();

    for (int iteration = 0;
         iteration < 3;
         ++iteration)
    {
        const AFPValue xy =
            multiplyValues(
                value,
                estimate);

        const AFPValue correction =
            subtractValues(
                two,
                xy);

        estimate =
            multiplyValues(
                estimate,
                correction);

        if (isZeroAFPValue(estimate))
            break;
    }

    return estimate;
}

AFPEncodedTensor AFPArithmetic::rms(
    const AFPEncodedTensor &input)
{
    if (input.size() == 0)
    {
        return buildTensorFromAFPValues(
            {zeroValue()},
            input.config_);
    }

    AFPAccumulator accumulator;

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        const AFPProduct square =
            multiplyAFPValues(
                value,
                value);

        accumulator =
            addProducts(
                accumulator,
                square);
    }

    if (accumulator.zero)
    {
        return buildTensorFromAFPValues(
            {zeroValue()},
            input.config_);
    }

    const int exponent =
        accumulator.exponent +
        integerLog2(
            static_cast<std::uint64_t>(
                accumulator.significand));

    AFPValue total =
        accumulatorToAFPValue(
            accumulator,
            exponent,
            false);

    std::size_t n =
        input.size();

    int log_n = 0;

    while ((std::size_t{1} << log_n) < n)
        ++log_n;

    total =
        scalePowerOfTwo(
            total,
            -log_n);

    const AFPValue result =
        sqrtValue(total);

    return buildTensorFromAFPValues(
        {result},
        input.config_);
}

AFPEncodedTensor AFPArithmetic::reciprocal(
    const AFPEncodedTensor &input)
{
    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        output.push_back(
            reciprocalValue(value));
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPArithmetic::AFPValue AFPArithmetic::sqrtValue(
    const AFPValue &value)
{
    if (isZeroAFPValue(value))
        return zeroValue();

    if (value.negative)
        throw std::invalid_argument(
            "AFP sqrt: negative input");

    const AFPProduct p =
        normalizeToProduct(value);

    const int top =
        p.scale_exponent +
        integerLog2(
            p.significand);

    const int initial_exponent =
        top / 2;

    AFPValue estimate;

    estimate.negative = false;

    estimate.exponent =
        static_cast<std::int8_t>(
            initial_exponent);

    estimate.offset = 0;

    estimate.mantissa =
        std::uint64_t{1} << 5;

    const AFPValue half =
        halfValue();

    for (int iteration = 0;
         iteration < 4;
         ++iteration)
    {
        const AFPValue quotient =
            divideValues(
                value,
                estimate);

        const AFPValue sum =
            addValues(
                estimate,
                quotient);

        estimate =
            multiplyValues(
                sum,
                half);
    }

    return estimate;
}

AFPEncodedTensor AFPArithmetic::sqrt(
    const AFPEncodedTensor &input)
{
    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        output.push_back(
            sqrtValue(value));
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::reciprocalSqrt(
    const AFPEncodedTensor &input)
{
    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        const AFPValue root =
            sqrtValue(value);

        output.push_back(
            reciprocalValue(root));
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::rmsNorm(
    const AFPEncodedTensor &input)
{
    if (input.size() == 0)
        return input;

    const AFPEncodedTensor rms_tensor =
        rms(input);

    const AFPValue rms_value =
        readAFPValue(
            rms_tensor,
            0,
            0);

    if (isZeroAFPValue(rms_value))
        return input;

    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        output.push_back(
            divideValues(
                value,
                rms_value));
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::layerNorm(
    const AFPEncodedTensor &input)
{
    if (input.size() == 0)
        return input;

    AFPAccumulator mean_accumulator;

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        const AFPProduct product =
            normalizeToProduct(value);

        mean_accumulator =
            addProducts(
                mean_accumulator,
                product);
    }

    if (mean_accumulator.zero)
    {
        return buildTensorFromAFPValues(
            std::vector<AFPValue>(
                input.size(),
                zeroValue()),
            input.config_);
    }

    const int mean_exponent =
        mean_accumulator.exponent +
        integerLog2(
            static_cast<std::uint64_t>(
                mean_accumulator.significand));

    AFPValue mean_value =
        accumulatorToAFPValue(
            mean_accumulator,
            mean_exponent,
            false);

    std::size_t n =
        input.size();

    int log_n = 0;

    while ((std::size_t{1} << log_n) < n)
        ++log_n;

    mean_value =
        scalePowerOfTwo(
            mean_value,
            -log_n);

    AFPAccumulator variance_accumulator;

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        const AFPValue centered =
            subtractValues(
                value,
                mean_value);

        const AFPProduct square =
            multiplyAFPValues(
                centered,
                centered);

        variance_accumulator =
            addProducts(
                variance_accumulator,
                square);
    }

    if (variance_accumulator.zero)
    {
        return buildTensorFromAFPValues(
            std::vector<AFPValue>(
                input.size(),
                zeroValue()),
            input.config_);
    }

    const int variance_exponent =
        variance_accumulator.exponent +
        integerLog2(
            static_cast<std::uint64_t>(
                variance_accumulator.significand));

    AFPValue variance =
        accumulatorToAFPValue(
            variance_accumulator,
            variance_exponent,
            false);

    variance =
        scalePowerOfTwo(
            variance,
            -log_n);

    AFPValue epsilon;

    epsilon.negative = false;
    epsilon.exponent = -10;
    epsilon.offset = 0;
    epsilon.mantissa =
        std::uint64_t{1} << 5;

    const AFPValue variance_with_epsilon =
        addValues(
            variance,
            epsilon);

    const AFPValue denominator =
        sqrtValue(
            variance_with_epsilon);

    if (isZeroAFPValue(denominator))
    {
        return buildTensorFromAFPValues(
            std::vector<AFPValue>(
                input.size(),
                zeroValue()),
            input.config_);
    }

    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t i = 0;
         i < input.size();
         ++i)
    {
        const AFPValue value =
            readAFPValue(
                input,
                i / block_size,
                i % block_size);

        const AFPValue centered =
            subtractValues(
                value,
                mean_value);

        output.push_back(
            divideValues(
                centered,
                denominator));
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::transpose(
    const AFPEncodedTensor &input,
    std::size_t rows,
    std::size_t columns)
{
    if (input.size() !=
        rows * columns)
    {
        throw std::invalid_argument(
            "AFP transpose: tensor size does not match dimensions");
    }

    std::vector<AFPValue> output;
    output.reserve(input.size());

    for (std::size_t column = 0;
         column < columns;
         ++column)
    {
        for (std::size_t row = 0;
             row < rows;
             ++row)
        {
            const std::size_t source_index =
                row * columns + column;

            output.push_back(
                readAFPValue(
                    input,
                    source_index / block_size,
                    source_index % block_size));
        }
    }

    return buildTensorFromAFPValues(
        output,
        input.config_);
}

AFPEncodedTensor AFPArithmetic::outerProduct(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    std::vector<AFPValue> output;

    output.reserve(
        a.size() *
        b.size());

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        const AFPValue av =
            readAFPValue(
                a,
                i / block_size,
                i % block_size);

        for (std::size_t j = 0;
             j < b.size();
             ++j)
        {
            const AFPValue bv =
                readAFPValue(
                    b,
                    j / block_size,
                    j % block_size);

            output.push_back(
                multiplyValues(
                    av,
                    bv));
        }
    }

    return buildTensorFromAFPValues(
        output,
        a.config_);
}

// ============================================================================
// Public Interface - Broadcasting Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::broadcastAdd(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b,
    std::size_t target_size)
{
    if (a.size() != target_size &&
        a.size() != 1)
    {
        throw std::invalid_argument(
            "AFP broadcast add: invalid first tensor size");
    }

    if (b.size() != target_size &&
        b.size() != 1)
    {
        throw std::invalid_argument(
            "AFP broadcast add: invalid second tensor size");
    }

    if (a.config_.block_size !=
        b.config_.block_size)
    {
        throw std::invalid_argument(
            "AFP block size mismatch");
    }

    AFPEncodedTensor result;

    result.config_ = a.config_;
    result.value_count_ = target_size;

    for (std::size_t block_start = 0;
         block_start < target_size;
         block_start += block_size)
    {
        result.block_offsets_.push_back(
            result.bits_.bitSize());

        int block_shared_exponent = -126;
        bool found_nonzero = false;

        for (std::size_t i = 0;
             i < block_size &&
             block_start + i < target_size;
             ++i)
        {
            const std::size_t a_idx =
                a.size() == 1 ?
                    0 :
                    block_start + i;

            const std::size_t b_idx =
                b.size() == 1 ?
                    0 :
                    block_start + i;

            const AFPValue a_val =
                readAFPValue(
                    a,
                    a_idx / block_size,
                    a_idx % block_size);

            const AFPValue b_val =
                readAFPValue(
                    b,
                    b_idx / block_size,
                    b_idx % block_size);

            if (!isZeroAFPValue(a_val))
            {
                const int a_eff_exp =
                    static_cast<int>(
                        a_val.exponent) -
                    static_cast<int>(
                        a_val.offset);

                if (!found_nonzero ||
                    a_eff_exp >
                        block_shared_exponent)
                {
                    block_shared_exponent =
                        a_eff_exp;

                    found_nonzero = true;
                }
            }

            if (!isZeroAFPValue(b_val))
            {
                const int b_eff_exp =
                    static_cast<int>(
                        b_val.exponent) -
                    static_cast<int>(
                        b_val.offset);

                if (!found_nonzero ||
                    b_eff_exp >
                        block_shared_exponent)
                {
                    block_shared_exponent =
                        b_eff_exp;

                    found_nonzero = true;
                }
            }
        }

        std::vector<AFPValue> block_values;

        for (std::size_t i = 0;
             i < block_size &&
             block_start + i < target_size;
             ++i)
        {
            const std::size_t actual_idx =
                block_start + i;

            const std::size_t a_idx =
                a.size() == 1 ?
                    0 :
                    actual_idx;

            const std::size_t b_idx =
                b.size() == 1 ?
                    0 :
                    actual_idx;

            const AFPValue a_val =
                readAFPValue(
                    a,
                    a_idx / block_size,
                    a_idx % block_size);

            const AFPValue b_val =
                readAFPValue(
                    b,
                    b_idx / block_size,
                    b_idx % block_size);

            const AFPAccumulator acc =
                addAFPValues(
                    a_val,
                    b_val);

            AFPValue result_val;

            result_val.negative =
                acc.negative;

            result_val.exponent =
                -126;

            result_val.offset =
                acc.zero ?
                    maximum_offset :
                    0;

            result_val.mantissa =
                acc.zero ?
                    0 :
                    32;

            block_values.push_back(
                result_val);
        }

        const bool first_half_positive =
            computeHalfPositive(
                block_values,
                0,
                true);

        const bool second_half_positive =
            computeHalfPositive(
                block_values,
                0,
                false);

        const std::uint8_t characterization =
            buildCharacterization(
                first_half_positive,
                second_half_positive);

        result.bits_.writeBits(
            encodeSharedExponent(
                block_shared_exponent),
            8);

        result.bits_.writeBits(
            characterization,
            8);

        for (std::size_t i = 0;
             i < block_size;
             ++i)
        {
            if (block_start + i >= target_size)
                break;

            const std::size_t a_idx =
                a.size() == 1 ?
                    0 :
                    block_start + i;

            const std::size_t b_idx =
                b.size() == 1 ?
                    0 :
                    block_start + i;

            const AFPValue a_val =
                readAFPValue(
                    a,
                    a_idx / block_size,
                    a_idx % block_size);

            const AFPValue b_val =
                readAFPValue(
                    b,
                    b_idx / block_size,
                    b_idx % block_size);

            const AFPAccumulator acc =
                addAFPValues(
                    a_val,
                    b_val);

            const bool positive_field =
                i < half_block_size ?
                    first_half_positive :
                    second_half_positive;

            const AFPValue afp_result =
                accumulatorToAFPValue(
                    acc,
                    block_shared_exponent,
                    positive_field);

            writeAFPValue(
                result.bits_,
                afp_result,
                positive_field);
        }
    }

    return result;
}

AFPEncodedTensor AFPArithmetic::broadcastMultiply(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b,
    std::size_t target_size)
{
    if (a.size() != target_size &&
        a.size() != 1)
    {
        throw std::invalid_argument(
            "AFP broadcast multiply: invalid first tensor size");
    }

    if (b.size() != target_size &&
        b.size() != 1)
    {
        throw std::invalid_argument(
            "AFP broadcast multiply: invalid second tensor size");
    }

    if (a.config_.block_size !=
        b.config_.block_size)
    {
        throw std::invalid_argument(
            "AFP block size mismatch");
    }

    AFPEncodedTensor result;

    result.config_ = a.config_;
    result.value_count_ = target_size;

    for (std::size_t block_start = 0;
         block_start < target_size;
         block_start += block_size)
    {
        result.block_offsets_.push_back(
            result.bits_.bitSize());

        int block_shared_exponent = -126;
        bool found_nonzero = false;

        for (std::size_t i = 0;
             i < block_size &&
             block_start + i < target_size;
             ++i)
        {
            const std::size_t a_idx =
                a.size() == 1 ?
                    0 :
                    block_start + i;

            const std::size_t b_idx =
                b.size() == 1 ?
                    0 :
                    block_start + i;

            const AFPValue a_val =
                readAFPValue(
                    a,
                    a_idx / block_size,
                    a_idx % block_size);

            const AFPValue b_val =
                readAFPValue(
                    b,
                    b_idx / block_size,
                    b_idx % block_size);

            const AFPProduct prod =
                multiplyAFPValues(
                    a_val,
                    b_val);

            if (!prod.zero)
            {
                const int prod_eff_exp =
                    prod.scale_exponent +
                    integerLog2(
                        prod.significand);

                if (!found_nonzero ||
                    prod_eff_exp >
                        block_shared_exponent)
                {
                    block_shared_exponent =
                        prod_eff_exp;

                    found_nonzero = true;
                }
            }
        }

        std::vector<AFPValue> block_values;

        for (std::size_t i = 0;
             i < block_size &&
             block_start + i < target_size;
             ++i)
        {
            const std::size_t a_idx =
                a.size() == 1 ?
                    0 :
                    block_start + i;

            const std::size_t b_idx =
                b.size() == 1 ?
                    0 :
                    block_start + i;

            const AFPValue a_val =
                readAFPValue(
                    a,
                    a_idx / block_size,
                    a_idx % block_size);

            const AFPValue b_val =
                readAFPValue(
                    b,
                    b_idx / block_size,
                    b_idx % block_size);

            const AFPProduct prod =
                multiplyAFPValues(
                    a_val,
                    b_val);

            AFPValue result_val;

            result_val.negative =
                prod.negative;

            result_val.exponent =
                -126;

            result_val.offset =
                prod.zero ?
                    maximum_offset :
                    0;

            result_val.mantissa =
                prod.zero ?
                    0 :
                    32;

            block_values.push_back(
                result_val);
        }

        const bool first_half_positive =
            computeHalfPositive(
                block_values,
                0,
                true);

        const bool second_half_positive =
            computeHalfPositive(
                block_values,
                0,
                false);

        const std::uint8_t characterization =
            buildCharacterization(
                first_half_positive,
                second_half_positive);

        result.bits_.writeBits(
            encodeSharedExponent(
                block_shared_exponent),
            8);

        result.bits_.writeBits(
            characterization,
            8);

        for (std::size_t i = 0;
             i < block_size;
             ++i)
        {
            if (block_start + i >= target_size)
                break;

            const std::size_t a_idx =
                a.size() == 1 ?
                    0 :
                    block_start + i;

            const std::size_t b_idx =
                b.size() == 1 ?
                    0 :
                    block_start + i;

            const AFPValue a_val =
                readAFPValue(
                    a,
                    a_idx / block_size,
                    a_idx % block_size);

            const AFPValue b_val =
                readAFPValue(
                    b,
                    b_idx / block_size,
                    b_idx % block_size);

            const AFPProduct prod =
                multiplyAFPValues(
                    a_val,
                    b_val);

            AFPAccumulator acc;

            acc.negative =
                prod.negative;

            acc.significand =
                static_cast<std::int64_t>(
                    prod.significand);

            acc.exponent =
                prod.scale_exponent;

            acc.zero =
                prod.zero;

            const bool positive_field =
                i < half_block_size ?
                    first_half_positive :
                    second_half_positive;

            const AFPValue afp_result =
                accumulatorToAFPValue(
                    acc,
                    block_shared_exponent,
                    positive_field);

            writeAFPValue(
                result.bits_,
                afp_result,
                positive_field);
        }
    }

    return result;
}