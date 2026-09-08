#include "../include/afp_math_new.hpp"
#include "../include/afp_thread_pool.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

/*
    AFP-NATIVE ARITHMETIC CONTRACT
    ------------------------------
    All operations in this file are computed in AFP representation:
    decode to AFP::Value, integer arithmetic on Product/Accumulator,
    re-encode via buildTensorFromAFPValues. Nothing converts AFP data
    to float/double for computation; float I/O lives only in the codec
    (AFPQuantizer::encode / AFPQuantizer::decode in
    afp_encoded_tensor_new.cpp).
*/

// Constants
constexpr std::size_t AFPArithmetic::block_size;
constexpr std::size_t AFPArithmetic::half_block_size;
constexpr int AFPArithmetic::maximum_offset;
constexpr int AFPArithmetic::fp32_exponent_bias;

// Helper function to create single-element vector
static std::vector<AFP::Value> makeSingleValue(AFP::Value value) {
    std::vector<AFP::Value> result;
    result.push_back(value);
    return result;
}

/*
    Divide total by the element count n for the mean/rms/layerNorm paths.

    The previous call sites used scalePowerOfTwo(-log2_ceil(n)), which
    divides by the next power of two: for non-power-of-two sizes (e.g.
    n = 100 -> 128) rms/layerNorm were off by up to ~37%. Dividing
    through Value::divide is exact in the AFP sense (one rounding step,
    24-bit division precision) and correct for every n. The count keeps
    a 5-bit mantissa, so n up to 255 encodes exactly.
*/
static AFP::Value divideByCount(const AFP::Value& total, std::size_t n) {
    AFP::Value count;
    count.negative = false;
    count.offset = 0;

    const int n_exponent = AFP::Utils::integerLog2(static_cast<uint64_t>(n));
    count.exponent = static_cast<int8_t>(n_exponent);

    const uint64_t normalized_n = static_cast<uint64_t>(n) << 5;
    count.mantissa = static_cast<uint8_t>(
        AFP::Utils::shiftRightRounded(normalized_n, n_exponent));
    if (count.mantissa == 0) count.mantissa = 1;

    return total.divide(count);
}

// ============================================================================
// AFP Value I/O Operations
// ============================================================================

AFP::Value AFPArithmetic::readAFPValue(
    const AFPEncodedTensor &tensor,
    std::size_t block_index,
    std::size_t value_index)
{
    constexpr std::size_t shared_exponent_bits = 8;
    constexpr std::size_t characterization_bits = 8;
    constexpr std::size_t first_field_bits = 1;
    constexpr std::size_t offset_bits = 3;
    constexpr std::size_t stored_mantissa_bits = 5;
    
    constexpr std::size_t block_header_bits = shared_exponent_bits + characterization_bits;
    constexpr std::size_t value_bits = first_field_bits + offset_bits + stored_mantissa_bits;
    
    constexpr uint8_t first_half_positive_bit = 0;
    constexpr uint8_t second_half_positive_bit = 1;
    
    const std::size_t block_base = tensor.block_offsets_[block_index];
    
    const uint8_t exponent_field = static_cast<uint8_t>(
        tensor.bits_.readBits(block_base, shared_exponent_bits));
    
    const int exponent = AFP::Utils::decodeSharedExponent(exponent_field);
    
    const uint8_t characterization = static_cast<uint8_t>(
        tensor.bits_.readBits(block_base + shared_exponent_bits, characterization_bits));
    
    const bool is_first_half = value_index < half_block_size;
    const bool positive_half = is_first_half ?
        (characterization & (uint8_t{1} << first_half_positive_bit)) != 0 :
        (characterization & (uint8_t{1} << second_half_positive_bit)) != 0;
    
    const std::size_t value_base = block_base + block_header_bits + value_index * value_bits;
    
    const uint8_t first_field = static_cast<uint8_t>(
        tensor.bits_.readBits(value_base, first_field_bits));
    
    const uint8_t offset = static_cast<uint8_t>(
        tensor.bits_.readBits(value_base + first_field_bits, offset_bits));
    
    uint8_t mantissa = static_cast<uint8_t>(
        tensor.bits_.readBits(value_base + first_field_bits + offset_bits, stored_mantissa_bits));
    
    AFP::Value result;
    result.exponent = static_cast<int8_t>(exponent);
    result.offset = offset;
    result.positive_field = positive_half;
    
    if (positive_half) {
        mantissa |= static_cast<uint8_t>(first_field << stored_mantissa_bits);
        result.negative = false;
        
        if (result.offset < maximum_offset) {
            result.mantissa = (uint64_t{1} << 6) + mantissa;
        } else {
            result.mantissa = mantissa;
        }
    } else {
        result.negative = first_field != 0;
        
        if (result.offset < maximum_offset) {
            result.mantissa = (uint64_t{1} << 5) + mantissa;
        } else {
            result.mantissa = mantissa;
        }
    }
    
    if (result.isZero()) result.negative = false;
    
    return result;
}

void AFPArithmetic::writeAFPValue(
    BitStream &bits,
    const AFP::Value &value,
    bool positive_field)
{
    constexpr std::size_t stored_mantissa_bits = 5;
    const int value_mantissa_bits = positive_field ? 6 : 5;
    
    uint64_t mantissa = 0;
    
    if (value.offset < maximum_offset) {
        const uint64_t implicit = uint64_t{1} << value_mantissa_bits;
        if (value.mantissa >= implicit) {
            mantissa = value.mantissa - implicit;
        }
    } else {
        mantissa = value.mantissa;
    }
    
    if (positive_field) {
        const uint64_t extra_bit = mantissa >> stored_mantissa_bits;
        const uint64_t stored_mantissa = mantissa & AFP::Utils::maskBits(stored_mantissa_bits);
        
        bits.writeBits(extra_bit, 1);
        bits.writeBits(static_cast<uint64_t>(value.offset), 3);
        bits.writeBits(stored_mantissa, stored_mantissa_bits);
    } else {
        bits.writeBits(value.negative ? 1 : 0, 1);
        bits.writeBits(static_cast<uint64_t>(value.offset), 3);
        bits.writeBits(mantissa & AFP::Utils::maskBits(stored_mantissa_bits), stored_mantissa_bits);
    }
}

// ============================================================================
// Bulk Block Decoding
// ============================================================================

/*
    Decode all block_size values of one block in a single pass.

    The block header (shared exponent + characterization) is read once
    instead of once per element, and the values are extracted in one
    ascending sweep of the bit stream. Bit-exact with readAFPValue for
    every element of the block.
*/

void AFPArithmetic::decodeBlock(
    const AFPEncodedTensor &tensor,
    std::size_t block_index,
    AFP::Value *out_values)
{
    constexpr std::size_t shared_exponent_bits = 8;
    constexpr std::size_t characterization_bits = 8;
    constexpr std::size_t first_field_bits = 1;
    constexpr std::size_t offset_bits = 3;
    constexpr std::size_t stored_mantissa_bits = 5;

    constexpr std::size_t block_header_bits =
        shared_exponent_bits + characterization_bits;

    constexpr std::size_t value_bits =
        first_field_bits + offset_bits + stored_mantissa_bits;

    constexpr uint8_t first_half_positive_bit = 0;
    constexpr uint8_t second_half_positive_bit = 1;

    const std::size_t block_base = tensor.block_offsets_[block_index];

    /*
        The whole block fits in 160 bits (20 bytes): read it once into
        three 64-bit windows and extract the 9-bit fields with shifts
        and masks. Bit-exact with the per-field readBits calls, but one
        bounds check and one memory sweep per block instead of
        forty-eight.
    */
    constexpr std::size_t block_total_bits = block_header_bits + block_size * value_bits;
    static_assert(block_total_bits <= 64 + 64 + 64,
                  "block decode assumes a three-window read");

    uint64_t w0 = 0, w1 = 0, w2 = 0;
    {
        const std::vector<uint8_t> &data = tensor.bits_.data();
        const std::size_t base_byte = block_base / 8;

        if (data.size() >= base_byte + 20) {
            for (std::size_t i = 0; i < 8; ++i) {
                w0 |= static_cast<uint64_t>(data[base_byte + i]) << (8 * i);
                w1 |= static_cast<uint64_t>(data[base_byte + 8 + i]) << (8 * i);
            }
            for (std::size_t i = 0; i < 4; ++i) {
                w2 |= static_cast<uint64_t>(data[base_byte + 16 + i]) << (8 * i);
            }
        } else {
            /*
                Truncated/malformed stream: fall back to the bounds-
                checked per-field reader so the error surfaces exactly
                as it did before this fast path existed.
            */
            for (std::size_t value_index = 0; value_index < block_size; ++value_index) {
                out_values[value_index] = readAFPValue(tensor, block_index, value_index);
            }
            return;
        }
    }

    const uint8_t exponent_field = static_cast<uint8_t>(w0 & 0xFF);
    const int exponent = AFP::Utils::decodeSharedExponent(exponent_field);

    const uint8_t characterization = static_cast<uint8_t>((w0 >> 8) & 0xFF);

    const bool first_half_positive =
        (characterization & (uint8_t{1} << first_half_positive_bit)) != 0;

    const bool second_half_positive =
        (characterization & (uint8_t{1} << second_half_positive_bit)) != 0;

    std::size_t bit_position = block_header_bits;

    for (std::size_t value_index = 0; value_index < block_size; ++value_index) {
        const bool positive_half =
            value_index < half_block_size ? first_half_positive : second_half_positive;

        /*
            Extract the 9-bit value field [bit_position, bit_position + 9)
            from the 144-bit window (w0 holds bits [0, 64), w1 [64, 128),
            w2 [128, 144)).
        */
        uint64_t field;
        if (bit_position < 56) {
            field = (w0 >> bit_position) & 0x1FF;
        } else if (bit_position < 64) {
            field = ((w0 >> bit_position) | (w1 << (64 - bit_position))) & 0x1FF;
        } else if (bit_position < 120) {
            field = (w1 >> (bit_position - 64)) & 0x1FF;
        } else if (bit_position < 128) {
            field = ((w1 >> (bit_position - 64)) | (w2 << (128 - bit_position))) & 0x1FF;
        } else {
            field = (w2 >> (bit_position - 128)) & 0x1FF;
        }
        bit_position += value_bits;

        const uint64_t first_field = field & 1;
        const uint64_t offset_field = (field >> first_field_bits) & 0x7;
        uint64_t mantissa = (field >> (first_field_bits + offset_bits)) & 0x1F;

        AFP::Value result;
        result.exponent = static_cast<int8_t>(exponent);
        result.offset = static_cast<uint8_t>(offset_field);
        result.positive_field = positive_half;

        if (positive_half) {
            mantissa |= first_field << stored_mantissa_bits;
            result.negative = false;

            if (result.offset < maximum_offset) {
                result.mantissa = static_cast<uint8_t>((uint64_t{1} << 6) + mantissa);
            } else {
                result.mantissa = static_cast<uint8_t>(mantissa);
            }
        } else {
            result.negative = first_field != 0;

            if (result.offset < maximum_offset) {
                result.mantissa = static_cast<uint8_t>((uint64_t{1} << 5) + mantissa);
            } else {
                result.mantissa = static_cast<uint8_t>(mantissa);
            }
        }

        if (result.isZero()) result.negative = false;

        out_values[value_index] = result;
    }
}

/*
    Decode a contiguous run of values starting at start_value.

    Sweeps forward through whole blocks, handling a partial first block
    when start_value is not block-aligned. Bit-exact with per-element
    readAFPValue calls.
*/

void AFPArithmetic::decodeRow(
    const AFPEncodedTensor &tensor,
    std::size_t start_value,
    std::size_t count,
    AFP::Value *out_values)
{
    std::size_t decoded = 0;

    AFP::Value block[block_size];

    std::size_t block_index = start_value / block_size;
    const std::size_t offset_in_block = start_value % block_size;

    if (offset_in_block != 0) {
        decodeBlock(tensor, block_index, block);

        while (decoded < count && offset_in_block + decoded < block_size) {
            out_values[decoded] = block[offset_in_block + decoded];
            ++decoded;
        }

        ++block_index;
    }

    while (decoded < count) {
        decodeBlock(tensor, block_index, block);

        const std::size_t take =
            (count - decoded < block_size) ? (count - decoded) : block_size;

        for (std::size_t i = 0; i < take; ++i) {
            out_values[decoded + i] = block[i];
        }

        decoded += take;
        ++block_index;
    }
}

// ============================================================================
// Block-Level Operations
// ============================================================================

int AFPArithmetic::computeSharedExponent(
    const std::vector<AFP::Value> &values,
    std::size_t block_start)
{
    int shared_exponent = -126;
    bool found_nonzero = false;
    
    for (std::size_t i = 0; i < block_size && block_start + i < values.size(); ++i) {
        const AFP::Value &value = values[block_start + i];
        if (value.isZero()) continue;
        
        const int effective_exponent = value.effectiveExponent();
        if (!found_nonzero || effective_exponent > shared_exponent) {
            shared_exponent = effective_exponent;
            found_nonzero = true;
        }
    }
    
    return shared_exponent;
}

bool AFPArithmetic::computeHalfPositive(
    const std::vector<AFP::Value> &values,
    std::size_t block_start,
    bool first_half)
{
    const std::size_t start = first_half ? 0 : half_block_size;
    const std::size_t end = first_half ? half_block_size : block_size;
    
    for (std::size_t i = start; i < end && block_start + i < values.size(); ++i) {
        const AFP::Value &value = values[block_start + i];
        if (!value.isZero() && value.negative) {
            return false;
        }
    }
    
    return true;
}

uint8_t AFPArithmetic::buildCharacterization(
    bool first_half_positive,
    bool second_half_positive)
{
    constexpr uint8_t first_half_positive_bit = 0;
    constexpr uint8_t second_half_positive_bit = 1;
    
    uint8_t characterization = 0;
    if (first_half_positive) {
        characterization |= (uint8_t{1} << first_half_positive_bit);
    }
    if (second_half_positive) {
        characterization |= (uint8_t{1} << second_half_positive_bit);
    }
    
    return characterization;
}

// ============================================================================
// Tensor Construction
// ============================================================================

AFPEncodedTensor AFPArithmetic::buildTensorFromAFPValues(
    const std::vector<AFP::Value> &values,
    const AFPConfig &config)
{
    AFPEncodedTensor result;
    result.config_ = config;
    result.value_count_ = values.size();
    
    if (values.empty()) return result;
    
    for (std::size_t block_start = 0; block_start < values.size(); block_start += block_size) {
        result.block_offsets_.push_back(result.bits_.bitSize());
        
        const int shared_exponent = computeSharedExponent(values, block_start);
        const bool first_half_positive = computeHalfPositive(values, block_start, true);
        const bool second_half_positive = computeHalfPositive(values, block_start, false);
        const uint8_t characterization = buildCharacterization(first_half_positive, second_half_positive);
        
        result.bits_.writeBits(AFP::Utils::encodeSharedExponent(shared_exponent), 8);
        result.bits_.writeBits(characterization, 8);
        
        /*
            Encode all 16 slots of the block into a 144-bit payload:
            each value contributes 9 bits (sign/extra, offset,
            mantissa). The payload is packed into three 48-bit groups
            (6 + 5 + 5 fields; every shift stays below 64) and written
            with three writeBits calls instead of forty-eight. The
            resulting bitstream is identical.
        */
        uint64_t g0 = 0;
        uint64_t g1 = 0;
        uint64_t g2 = 0;

        for (std::size_t i = 0; i < block_size; ++i) {
            const bool positive_field = i < half_block_size ? first_half_positive : second_half_positive;
            
            AFP::Value value;
            if (block_start + i < values.size()) {
                const AFP::Product normalized = values[block_start + i].toProduct();
                AFP::Accumulator acc;
                acc.negative = normalized.negative;
                acc.significand = static_cast<int64_t>(normalized.significand);
                acc.exponent = normalized.scale_exponent;
                acc.zero = normalized.zero;
                value = AFP::Value::fromAccumulator(acc, shared_exponent, positive_field);
            } else {
                value.exponent = static_cast<int8_t>(shared_exponent);
                value.negative = false;
                value.offset = maximum_offset;
                value.mantissa = 0;
            }
            
            int field_mantissa_bits = positive_field ? 6 : 5;
            
            uint64_t mantissa = 0;
            
            if (value.offset < maximum_offset) {
                const uint64_t implicit = uint64_t{1} << field_mantissa_bits;
                if (value.mantissa >= implicit) {
                    mantissa = value.mantissa - implicit;
                }
            } else {
                mantissa = value.mantissa;
            }
            
            uint64_t first_field;
            if (positive_field) {
                first_field = mantissa >> 5;
                mantissa &= 0x1F;
            } else {
                first_field = value.negative ? 1 : 0;
                mantissa &= 0x1F;
            }
            
            const uint64_t field = first_field |
                (value.offset << 1) |
                (mantissa << 4);
            
            if (i < 6) {
                g0 |= field << (9 * i);
            } else if (i < 11) {
                g1 |= field << (9 * (i - 6));
            } else {
                g2 |= field << (9 * (i - 11));
            }
        }
        
        result.bits_.writeBits(g0, 54);
        result.bits_.writeBits(g1, 45);
        result.bits_.writeBits(g2, 45);
    }
    
    return result;
}

// ============================================================================
// Utilities
// ============================================================================

/*
    Accumulate one product into a running accumulator.

    This is the accumulation logic previously duplicated inside
    dotProduct, matrixVectorMultiply, matrixMultiply, softmax, sum,
    sumSquares and layerNorm; centralizing it keeps those routines
    readable and guarantees identical rounding everywhere.

    All accumulation now runs through the single saturating helper in
    afp_value.cpp (Utils::accumulateProduct), which is well-defined
    for every exponent gap: the previous inline int64 shifts silently
    mis-scaled a term whenever the gap to the common exponent reached
    63 bits (the shift was skipped and the term added at its own
    scale). For ordinary data the results are bit-identical - the
    helper's fast path performs the same alignment and rounding.
*/

void AFPArithmetic::accumulateProduct(
    AFP::Accumulator &accumulator,
    const AFP::Product &product)
{
    AFP::Utils::accumulateProduct(accumulator, product);
}

void AFPArithmetic::validateCompatible(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("AFP tensor size mismatch");
    }
    if (a.config_.block_size != b.config_.block_size) {
        throw std::invalid_argument("AFP block size mismatch");
    }
}

// ============================================================================
// Public Interface - Element-wise Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::add(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.add(bv);
    });
}

AFPEncodedTensor AFPArithmetic::subtract(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.subtract(bv);
    });
}

AFPEncodedTensor AFPArithmetic::multiply(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.multiply(bv);
    });
}

AFPEncodedTensor AFPArithmetic::divide(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.divide(bv);
    });
}

AFPEncodedTensor AFPArithmetic::negate(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.negate();
    });
}

AFPEncodedTensor AFPArithmetic::absolute(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.absolute();
    });
}

AFPEncodedTensor AFPArithmetic::minimum(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.compare(bv) <= 0 ? av : bv;
    });
}

AFPEncodedTensor AFPArithmetic::maximum(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    return elementWiseOp(a, b, [](const AFP::Value& av, const AFP::Value& bv) {
        return av.compare(bv) >= 0 ? av : bv;
    });
}

AFPEncodedTensor AFPArithmetic::clamp(
    const AFPEncodedTensor &input,
    const AFPEncodedTensor &min_value,
    const AFPEncodedTensor &max_value)
{
    if (min_value.size() != 1 || max_value.size() != 1) {
        throw std::invalid_argument("AFP clamp: min_value and max_value must contain one value");
    }
    
    const AFP::Value min_val = readAFPValue(min_value, 0, 0);
    const AFP::Value max_val = readAFPValue(max_value, 0, 0);
    
    if (min_val.compare(max_val) > 0) {
        throw std::invalid_argument("AFP clamp: min_value greater than max_value");
    }
    
    return elementWiseUnaryOp(input, [&](const AFP::Value& v) {
        if (v.compare(min_val) < 0) return min_val;
        if (v.compare(max_val) > 0) return max_val;
        return v;
    });
}

AFPEncodedTensor AFPArithmetic::leakyRelu(
    const AFPEncodedTensor &input,
    const AFPEncodedTensor &alpha)
{
    if (alpha.size() != 1) {
        throw std::invalid_argument("AFP leakyRelu: alpha must contain one value");
    }
    
    const AFP::Value alpha_value = readAFPValue(alpha, 0, 0);
    
    return elementWiseUnaryOp(input, [&](const AFP::Value& v) {
        if (v.isZero() || !v.negative) return v;
        return v.multiply(alpha_value);
    });
}

AFPEncodedTensor AFPArithmetic::silu(const AFPEncodedTensor &input)
{
    return multiply(input, sigmoid(input));
}

// ============================================================================
// Public Interface - Dot Product
// ============================================================================

AFPEncodedTensor AFPArithmetic::dotProduct(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    validateCompatible(a, b);
    
    if (a.size() == 0) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), a.config_);
    }
    
    /*
        Decode both operands once (row-at-a-time) instead of once per
        element pair; accumulation goes through the shared helper.
    */

    AFP::Accumulator accumulator;

    std::vector<AFP::Value> a_values(a.size());
    std::vector<AFP::Value> b_values(b.size());

    decodeRow(a, 0, a.size(), a_values.data());
    decodeRow(b, 0, b.size(), b_values.data());

    for (std::size_t index = 0; index < a.size(); ++index) {
        const AFP::Product product = a_values[index].toProduct();
        const AFP::Product prod_b = b_values[index].toProduct();

        AFP::Product combined;
        combined.negative = product.negative ^ prod_b.negative;
        combined.significand = product.significand * prod_b.significand;
        combined.scale_exponent = product.scale_exponent + prod_b.scale_exponent;
        combined.zero = combined.significand == 0;

        accumulateProduct(accumulator, combined);
    }
    
    if (accumulator.zero) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), a.config_);
    }
    
    const int exponent = accumulator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(accumulator.significand));
    return buildTensorFromAFPValues(makeSingleValue(AFP::Value::fromAccumulator(accumulator, exponent, false)), a.config_);
}

// ============================================================================
// Prepared Matrices and Unpacked-Value Kernels
// ============================================================================

/*
    Storage behind AFPPreparedMatrix: the full matrix as AFP::Product
    entries, decoded once from the AFP bitstream. Products are exactly
    what Value::toProduct() returns for the same weight, so kernels on
    the prepared form are bit-exact with the decode-every-call path.
*/

struct AFPPreparedMatrix::Impl {
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::vector<AFP::Product> products;

    const AFP::Product &at(std::size_t row, std::size_t column) const
    {
        return products[row * columns + column];
    }
};

AFPPreparedMatrix::AFPPreparedMatrix() : impl_(std::make_unique<Impl>()) {}
AFPPreparedMatrix::~AFPPreparedMatrix() = default;
AFPPreparedMatrix::AFPPreparedMatrix(AFPPreparedMatrix &&) noexcept = default;
AFPPreparedMatrix &AFPPreparedMatrix::operator=(AFPPreparedMatrix &&) noexcept = default;

std::size_t AFPPreparedMatrix::rows() const { return impl_->rows; }
std::size_t AFPPreparedMatrix::columns() const { return impl_->columns; }

const AFP::Product &AFPPreparedMatrix::product(std::size_t row, std::size_t column) const
{
    return impl_->at(row, column);
}

std::unique_ptr<AFPPreparedMatrix> AFPArithmetic::prepareMatrix(
    const AFPEncodedTensor &weights,
    std::size_t rows,
    std::size_t columns)
{
    if (weights.size() != rows * columns) {
        throw std::invalid_argument("AFP prepareMatrix: weight tensor size doesn't match dimensions");
    }

    auto prepared = std::make_unique<AFPPreparedMatrix>();
    prepared->impl_->rows = rows;
    prepared->impl_->columns = columns;
    prepared->impl_->products.resize(rows * columns);

    /*
        Decode all rows once. Each entry stores the exact Product
        (significand, scale_exponent, sign) that toProduct() returns,
        so accumulation on the prepared form is bit-identical to the
        decode-every-call implementation.
    */
    AFP::Value row[block_size];
    for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t base = r * columns;
        std::size_t done = 0;
        while (done < columns) {
            const std::size_t take =
                (columns - done < block_size) ? (columns - done) : block_size;
            decodeRow(weights, base + done, take, row);
            for (std::size_t i = 0; i < take; ++i) {
                prepared->impl_->products[base + done + i] = row[i].toProduct();
            }
            done += take;
        }
    }

    return prepared;
}

/*
    Shared multiply-accumulate core for both prepared entry points.

    Input products are converted once per call (hoisted out of the row
    loop - previously each input was reconverted once per output row).
    Row parallelism runs through the process-wide thread pool, so the
    parallelism stays inside the tensor operation without spawning
    threads per call and without oversubscribing a parallel caller.
*/

void AFPArithmetic::preparedMatVecKernel(
    const AFPPreparedMatrix &prepared,
    const AFP::Product *input_products,
    AFP::Value *output)
{
    const std::size_t rows = prepared.rows();
    const std::size_t columns = prepared.columns();

    const auto compute_range = [&](std::size_t first_row, std::size_t last_row) {
        for (std::size_t row = first_row; row < last_row; ++row) {
            AFP::Accumulator row_accumulator;

            for (std::size_t col = 0; col < columns; ++col) {
                const AFP::Product &product = prepared.product(row, col);
                const AFP::Product &prod_input = input_products[col];

                AFP::Product combined;
                combined.negative = product.negative ^ prod_input.negative;
                combined.significand = product.significand * prod_input.significand;
                combined.scale_exponent = product.scale_exponent + prod_input.scale_exponent;
                combined.zero = combined.significand == 0;

                accumulateProduct(row_accumulator, combined);
            }

            if (row_accumulator.zero) {
                output[row] = AFP::Value::zero();
            } else {
                const int exponent =
                    row_accumulator.exponent +
                    AFP::Utils::integerLog2(
                        static_cast<uint64_t>(row_accumulator.significand));

                output[row] =
                    AFP::Value::fromAccumulator(row_accumulator, exponent, false);
            }
        }
    };

    AFP::ThreadPool::parallelFor(0, rows, 8, compute_range);
}

AFPEncodedTensor AFPArithmetic::matrixVectorMultiplyPrepared(
    const AFPPreparedMatrix &prepared,
    const AFPEncodedTensor &input)
{
    if (input.size() != prepared.columns()) {
        throw std::invalid_argument("AFP prepared matvec: input size doesn't match columns");
    }

    std::vector<AFP::Value> input_values(prepared.columns());
    decodeRow(input, 0, prepared.columns(), input_values.data());

    std::vector<AFP::Product> input_products(prepared.columns());
    for (std::size_t col = 0; col < prepared.columns(); ++col) {
        input_products[col] = input_values[col].toProduct();
    }

    std::vector<AFP::Value> output(prepared.rows());
    preparedMatVecKernel(prepared, input_products.data(), output.data());

    return buildTensorFromAFPValues(output, input.config_);
}

std::vector<AFP::Value> AFPArithmetic::matrixVectorMultiplyUnpacked(
    const AFPPreparedMatrix &prepared,
    const std::vector<AFP::Value> &input)
{
    if (input.size() != prepared.columns()) {
        throw std::invalid_argument("AFP prepared matvec: input size doesn't match columns");
    }

    std::vector<AFP::Product> input_products(prepared.columns());
    for (std::size_t col = 0; col < prepared.columns(); ++col) {
        input_products[col] = input[col].toProduct();
    }

    std::vector<AFP::Value> output(prepared.rows());
    preparedMatVecKernel(prepared, input_products.data(), output.data());
    return output;
}

std::vector<AFP::Value> AFPArithmetic::addUnpacked(
    const std::vector<AFP::Value> &a,
    const std::vector<AFP::Value> &b)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("AFP addUnpacked: size mismatch");
    }

    std::vector<AFP::Value> output(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        output[i] = a[i].add(b[i]);
    }
    return output;
}

std::vector<AFP::Value> AFPArithmetic::reluUnpacked(const std::vector<AFP::Value> &input)
{
    std::vector<AFP::Value> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i].relu();
    }
    return output;
}

AFPEncodedTensor AFPArithmetic::encodeAFPValues(
    const std::vector<AFP::Value> &values,
    const AFPConfig &config)
{
    return buildTensorFromAFPValues(values, config);
}

std::vector<AFP::Value> AFPArithmetic::decodeAFPValues(const AFPEncodedTensor &tensor)
{
    std::vector<AFP::Value> values(tensor.size());
    if (!values.empty()) {
        decodeRow(tensor, 0, tensor.size(), values.data());
    }
    return values;
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
    if (weights.size() != rows * columns) {
        throw std::invalid_argument("Weight tensor size doesn't match dimensions");
    }
    if (input.size() != columns) {
        throw std::invalid_argument("Input tensor size doesn't match columns");
    }
    
    /*
        Decode the input vector once, hoist the input products out of
        the row loop (each input was previously reconverted once per
        output row), and run rows through the shared thread pool.

        Each row writes to a distinct output slot, so no locking is
        needed. Bit-exact with the sequential implementation because
        the per-row accumulation order is unchanged.
    */

    std::vector<AFP::Value> input_values(columns);
    decodeRow(input, 0, columns, input_values.data());

    std::vector<AFP::Product> input_products(columns);
    for (std::size_t col = 0; col < columns; ++col) {
        input_products[col] = input_values[col].toProduct();
    }

    std::vector<AFP::Value> output(rows);

    const auto compute_range = [&](std::size_t first_row, std::size_t last_row) {
        std::vector<AFP::Value> weight_row(columns);

        for (std::size_t row = first_row; row < last_row; ++row) {
            decodeRow(weights, row * columns, columns, weight_row.data());

            AFP::Accumulator row_accumulator;

            for (std::size_t col = 0; col < columns; ++col) {
                const AFP::Product product = weight_row[col].toProduct();
                const AFP::Product &prod_input = input_products[col];

                AFP::Product combined;
                combined.negative = product.negative ^ prod_input.negative;
                combined.significand = product.significand * prod_input.significand;
                combined.scale_exponent = product.scale_exponent + prod_input.scale_exponent;
                combined.zero = combined.significand == 0;

                accumulateProduct(row_accumulator, combined);
            }

            if (row_accumulator.zero) {
                output[row] = AFP::Value::zero();
            } else {
                const int exponent =
                    row_accumulator.exponent +
                    AFP::Utils::integerLog2(
                        static_cast<uint64_t>(row_accumulator.significand));

                output[row] =
                    AFP::Value::fromAccumulator(row_accumulator, exponent, false);
            }
        }
    };

    AFP::ThreadPool::parallelFor(0, rows, 8, compute_range);

    return buildTensorFromAFPValues(output, weights.config_);
}

AFPEncodedTensor AFPArithmetic::matrixMultiply(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b,
    std::size_t rows_a,
    std::size_t cols_a,
    std::size_t cols_b)
{
    if (a.size() != rows_a * cols_a) {
        throw std::invalid_argument("Matrix A size doesn't match dimensions");
    }
    if (b.size() != cols_a * cols_b) {
        throw std::invalid_argument("Matrix B size doesn't match dimensions");
    }
    
    /*
        Decode B once (it is shared by every output row), hoist the B
        products out of the inner loop (each B element was previously
        reconverted once per output row), and run output rows through
        the shared thread pool.

        Each row writes to a distinct row of the output, so no locking
        is needed. Bit-exact with the sequential implementation because
        the per-element accumulation order is unchanged.
    */

    std::vector<AFP::Value> b_values(cols_a * cols_b);
    decodeRow(b, 0, cols_a * cols_b, b_values.data());

    std::vector<AFP::Product> b_products(cols_a * cols_b);
    for (std::size_t idx = 0; idx < cols_a * cols_b; ++idx) {
        b_products[idx] = b_values[idx].toProduct();
    }

    std::vector<AFP::Value> output(rows_a * cols_b);

    const auto compute_range = [&](std::size_t first_row, std::size_t last_row) {
        std::vector<AFP::Value> a_row(cols_a);
        std::vector<AFP::Product> a_products(cols_a);

        for (std::size_t i = first_row; i < last_row; ++i) {
            decodeRow(a, i * cols_a, cols_a, a_row.data());
            for (std::size_t k = 0; k < cols_a; ++k) {
                a_products[k] = a_row[k].toProduct();
            }

            for (std::size_t j = 0; j < cols_b; ++j) {
                AFP::Accumulator element_accumulator;

                for (std::size_t k = 0; k < cols_a; ++k) {
                    const AFP::Product &product = a_products[k];
                    const AFP::Product &prod_b = b_products[k * cols_b + j];

                    AFP::Product combined;
                    combined.negative = product.negative ^ prod_b.negative;
                    combined.significand = product.significand * prod_b.significand;
                    combined.scale_exponent = product.scale_exponent + prod_b.scale_exponent;
                    combined.zero = combined.significand == 0;

                    accumulateProduct(element_accumulator, combined);
                }

                const std::size_t out_idx = i * cols_b + j;

                if (element_accumulator.zero) {
                    output[out_idx] = AFP::Value::zero();
                } else {
                    const int exponent =
                        element_accumulator.exponent +
                        AFP::Utils::integerLog2(
                            static_cast<uint64_t>(element_accumulator.significand));

                    output[out_idx] =
                        AFP::Value::fromAccumulator(element_accumulator, exponent, false);
                }
            }
        }
    };

    AFP::ThreadPool::parallelFor(0, rows_a, 1, compute_range);

    return buildTensorFromAFPValues(output, a.config_);
}

// ============================================================================
// Public Interface - Activation Functions
// ============================================================================

AFPEncodedTensor AFPArithmetic::relu(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.relu();
    });
}

AFPEncodedTensor AFPArithmetic::sigmoid(const AFPEncodedTensor &input)
{
    /*
        Sigmoid via the exponential definition, 1 / (1 + e^-x):

        - x >= 0: computed directly through the (now correct) exp();
          for large x the denominator rounds to 1 and the result
          saturates at 1.
        - x < 0: computed as e^x / (1 + e^x), which stays accurate for
          every negative argument (the previous cubic Taylor
          approximation 0.5 + 0.25x - x^3/48 diverged for x < -1:
          sigmoid(-10) returned ~19 instead of ~0).
        - x == 0: exact 0.5.
    */
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        if (v.isZero()) {
            AFP::Value result;
            result.negative = false;
            result.exponent = -1;
            result.offset = 0;
            result.mantissa = 32;
            return result;
        }

        const AFP::Value one = AFP::Value::one();

        if (!v.isNegative()) {
            const AFP::Value negated = v.negate();
            const AFP::Value denominator = one.add(negated.exp());
            return one.divide(denominator);
        }

        const AFP::Value e = v.exp();
        const AFP::Value denominator = one.add(e);
        return e.divide(denominator);
    });
}

AFPEncodedTensor AFPArithmetic::tanh(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.tanh();
    });
}

AFPEncodedTensor AFPArithmetic::gelu(const AFPEncodedTensor &input)
{
    const AFP::Value half = AFP::Value::half();
    
    AFP::Value three_quarters;
    three_quarters.negative = false;
    three_quarters.exponent = -1;
    three_quarters.offset = 0;
    three_quarters.mantissa = (uint64_t{1} << 5) + 16;
    
    AFP::Value one_sixteenth;
    one_sixteenth.negative = false;
    one_sixteenth.exponent = -4;
    one_sixteenth.offset = 0;
    one_sixteenth.mantissa = uint64_t{1} << 5;
    
    const AFP::Value one = AFP::Value::one();
    
    return elementWiseUnaryOp(input, [&](const AFP::Value& x) {
        const AFP::Value x2 = x.multiply(x);
        const AFP::Value x3 = x2.multiply(x);
        const AFP::Value cubic = x3.multiply(one_sixteenth);
        const AFP::Value inner = x.add(cubic);
        const AFP::Value scaled = inner.multiply(three_quarters);
        const AFP::Value t = scaled.tanh();
        const AFP::Value one_plus_t = one.add(t);
        const AFP::Value half_x = x.multiply(half);
        return half_x.multiply(one_plus_t);
    });
}

AFPEncodedTensor AFPArithmetic::exp(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.exp();
    });
}

AFPEncodedTensor AFPArithmetic::softmax(const AFPEncodedTensor &input)
{
    if (input.size() == 0) return input;
    
    /*
        Decode once, reuse for both the max scan and the exponentials.
    */

    std::vector<AFP::Value> input_values(input.size());
    decodeRow(input, 0, input.size(), input_values.data());

    AFP::Value maximum_value = input_values[0];
    for (std::size_t i = 1; i < input.size(); ++i) {
        if (input_values[i].compare(maximum_value) > 0) {
            maximum_value = input_values[i];
        }
    }
    
    std::vector<AFP::Value> exponentials;
    exponentials.reserve(input.size());
    
    AFP::Accumulator denominator;
    
    for (std::size_t i = 0; i < input.size(); ++i) {
        const AFP::Value shifted = input_values[i].subtract(maximum_value);
        const AFP::Value e = shifted.exp();
        exponentials.push_back(e);
        
        accumulateProduct(denominator, e.toProduct());
    }
    
    if (denominator.zero) {
        return buildTensorFromAFPValues(std::vector<AFP::Value>(input.size(), AFP::Value::zero()), input.config_);
    }
    
    const int denominator_exponent = denominator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(denominator.significand));
    const AFP::Value denominator_value = AFP::Value::fromAccumulator(denominator, denominator_exponent, false);
    
    std::vector<AFP::Value> output;
    output.reserve(input.size());
    
    for (const AFP::Value& e : exponentials) {
        output.push_back(e.divide(denominator_value));
    }
    
    return buildTensorFromAFPValues(output, input.config_);
}

// ============================================================================
// Public Interface - Reduction Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::sum(const AFPEncodedTensor &input)
{
    if (input.size() == 0) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    AFP::Accumulator accumulator;

    std::vector<AFP::Value> input_values(input.size());
    decodeRow(input, 0, input.size(), input_values.data());

    for (std::size_t index = 0; index < input.size(); ++index) {
        accumulateProduct(accumulator, input_values[index].toProduct());
    }
    
    if (accumulator.zero) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    const int exponent = accumulator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(accumulator.significand));
    return buildTensorFromAFPValues(std::vector<AFP::Value>{AFP::Value::fromAccumulator(accumulator, exponent, false)}, input.config_);
}

AFPEncodedTensor AFPArithmetic::mean(const AFPEncodedTensor &input)
{
    if (input.size() == 0) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    const AFPEncodedTensor sum_tensor = sum(input);
    const AFP::Value sum_value = readAFPValue(sum_tensor, 0, 0);

    /*
        Divide by n exactly: correct for every n (the count encoding
        below keeps 5+ mantissa bits, so n up to 255 is exact).
    */
    const AFP::Value mean_value = divideByCount(sum_value, input.size());
    return buildTensorFromAFPValues(makeSingleValue(mean_value), input.config_);
}

AFPEncodedTensor AFPArithmetic::max(const AFPEncodedTensor &input)
{
    if (input.size() == 0) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    std::vector<AFP::Value> input_values(input.size());
    decodeRow(input, 0, input.size(), input_values.data());

    AFP::Value max_value = input_values[0];
    for (std::size_t i = 1; i < input.size(); ++i) {
        if (input_values[i].compare(max_value) > 0) {
            max_value = input_values[i];
        }
    }
    
    return buildTensorFromAFPValues(makeSingleValue(max_value), input.config_);
}

AFPEncodedTensor AFPArithmetic::sumSquares(const AFPEncodedTensor &input)
{
    AFP::Accumulator accumulator;

    std::vector<AFP::Value> input_values(input.size());
    decodeRow(input, 0, input.size(), input_values.data());

    for (std::size_t i = 0; i < input.size(); ++i) {
        const AFP::Product square = input_values[i].toProduct();
        
        AFP::Product squared;
        squared.negative = false;
        squared.significand = square.significand * square.significand;
        squared.scale_exponent = square.scale_exponent * 2;
        squared.zero = squared.significand == 0;
        
        accumulateProduct(accumulator, squared);
    }
    
    if (accumulator.zero) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    const int exponent = accumulator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(accumulator.significand));
    return buildTensorFromAFPValues(std::vector<AFP::Value>{AFP::Value::fromAccumulator(accumulator, exponent, false)}, input.config_);
}

AFPEncodedTensor AFPArithmetic::rms(const AFPEncodedTensor &input)
{
    if (input.size() == 0) {
        return buildTensorFromAFPValues(makeSingleValue(AFP::Value::zero()), input.config_);
    }
    
    const AFPEncodedTensor sum_squares_tensor = sumSquares(input);
    const AFP::Value total = readAFPValue(sum_squares_tensor, 0, 0);

    /*
        Divide by n exactly: scalePowerOfTwo(-log_n) divided by the next
        power of two, inflating the rms by up to ~37% for non-power-of-two
        sizes (e.g. n = 100 -> divided by 128).
    */
    const AFP::Value scaled = divideByCount(total, input.size());
    const AFP::Value result = scaled.sqrt();
    
    return buildTensorFromAFPValues(makeSingleValue(result), input.config_);
}

// ============================================================================
// Public Interface - Element-wise Mathematical Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::reciprocal(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.reciprocal();
    });
}

AFPEncodedTensor AFPArithmetic::sqrt(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.sqrt();
    });
}

AFPEncodedTensor AFPArithmetic::reciprocalSqrt(const AFPEncodedTensor &input)
{
    return elementWiseUnaryOp(input, [](const AFP::Value& v) {
        return v.reciprocalSqrt();
    });
}

// ============================================================================
// Public Interface - Normalization
// ============================================================================

AFPEncodedTensor AFPArithmetic::rmsNorm(const AFPEncodedTensor &input)
{
    if (input.size() == 0) return input;
    
    const AFPEncodedTensor rms_tensor = rms(input);
    const AFP::Value rms_value = readAFPValue(rms_tensor, 0, 0);
    
    if (rms_value.isZero()) return input;
    
    return elementWiseUnaryOp(input, [&](const AFP::Value& v) {
        return v.divide(rms_value);
    });
}

AFPEncodedTensor AFPArithmetic::layerNorm(const AFPEncodedTensor &input)
{
    if (input.size() == 0) return input;
    
    const AFPEncodedTensor mean_tensor = mean(input);
    const AFP::Value mean_value = readAFPValue(mean_tensor, 0, 0);

    /*
        mean() already divides by n exactly; the previous extra
        scalePowerOfTwo(-log_n) rescaled the mean again by 2^-ceil(log2 n),
        double-counting the division for non-power-of-two sizes.
    */
    const AFP::Value scaled_mean = mean_value;

    std::vector<AFP::Value> input_values(input.size());
    decodeRow(input, 0, input.size(), input_values.data());

    AFP::Accumulator variance_accumulator;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const AFP::Value centered = input_values[i].subtract(scaled_mean);
        const AFP::Product square = centered.toProduct();
        
        AFP::Product squared;
        squared.negative = false;
        squared.significand = square.significand * square.significand;
        squared.scale_exponent = square.scale_exponent * 2;
        squared.zero = squared.significand == 0;
        
        accumulateProduct(variance_accumulator, squared);
    }
    
    if (variance_accumulator.zero) {
        return buildTensorFromAFPValues(std::vector<AFP::Value>(input.size(), AFP::Value::zero()), input.config_);
    }
    
    const int variance_exponent = variance_accumulator.exponent + AFP::Utils::integerLog2(static_cast<uint64_t>(variance_accumulator.significand));
    AFP::Value variance = AFP::Value::fromAccumulator(variance_accumulator, variance_exponent, false);

    /*
        Divide by n exactly (see rms): the previous power-of-two
        rescale inflated the variance by up to ~37%.
    */
    variance = divideByCount(variance, input.size());
    
    AFP::Value epsilon;
    epsilon.negative = false;
    epsilon.exponent = -10;
    epsilon.offset = 0;
    epsilon.mantissa = uint64_t{1} << 5;
    
    const AFP::Value variance_with_epsilon = variance.add(epsilon);
    const AFP::Value denominator = variance_with_epsilon.sqrt();
    
    if (denominator.isZero()) {
        return buildTensorFromAFPValues(std::vector<AFP::Value>(input.size(), AFP::Value::zero()), input.config_);
    }
    
    return elementWiseUnaryOp(input, [&](const AFP::Value& v) {
        const AFP::Value centered = v.subtract(scaled_mean);
        return centered.divide(denominator);
    });
}

// ============================================================================
// Public Interface - Tensor Shape Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::transpose(
    const AFPEncodedTensor &input,
    std::size_t rows,
    std::size_t columns)
{
    if (input.size() != rows * columns) {
        throw std::invalid_argument("AFP transpose: tensor size does not match dimensions");
    }
    
    std::vector<AFP::Value> output;
    output.reserve(input.size());
    
    /*
        Decode each source block once and scatter its 16 values to
        their transposed positions. Previously every element paid a
        per-value readAFPValue call (three bounds-checked bit reads);
        for a wide row (columns >> block_size) the same block was
        re-decoded once per output row, so this also removes the
        redundant block re-parsing.
    */
    std::vector<AFP::Value> block(block_size);
    
    for (std::size_t block_index = 0; block_index * block_size < input.size(); ++block_index) {
        decodeBlock(input, block_index, block.data());
        
        const std::size_t block_start = block_index * block_size;
        for (std::size_t i = 0; i < block_size && block_start + i < input.size(); ++i) {
            const std::size_t source_index = block_start + i;
            const std::size_t row = source_index / columns;
            const std::size_t column = source_index % columns;
            output[column * rows + row] = block[i];
        }
    }
    
    return buildTensorFromAFPValues(output, input.config_);
}

AFPEncodedTensor AFPArithmetic::outerProduct(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    std::vector<AFP::Value> output;
    output.reserve(a.size() * b.size());
    
    /*
        Decode both operands once up front (block-at-a-time) instead of
        re-reading b's blocks once per a element: the inner loop used
        to re-parse the same 16-value block of b for every outer value.
    */
    const std::vector<AFP::Value> a_values = decodeAFPValues(a);
    const std::vector<AFP::Value> b_values = decodeAFPValues(b);
    
    for (std::size_t i = 0; i < a.size(); ++i) {
        const AFP::Value &av = a_values[i];
        for (std::size_t j = 0; j < b.size(); ++j) {
            output.push_back(av.multiply(b_values[j]));
        }
    }
    
    return buildTensorFromAFPValues(output, a.config_);
}

// ============================================================================
// Public Interface - Broadcasting Operations
// ============================================================================

AFPEncodedTensor AFPArithmetic::broadcastAdd(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b,
    std::size_t target_size)
{
    if ((a.size() != target_size && a.size() != 1) || (b.size() != target_size && b.size() != 1)) {
        throw std::invalid_argument("AFP broadcast add: invalid tensor sizes");
    }
    if (a.config_.block_size != b.config_.block_size) {
        throw std::invalid_argument("AFP block size mismatch");
    }
    
    std::vector<AFP::Value> output;
    output.reserve(target_size);
    
    /*
        Decode each operand once (block-at-a-time) and reuse the
        decoded values across the broadcast loop; broadcast operands
        are usually small (bias-like), so this also caps the extra
        decode memory.
    */
    const std::vector<AFP::Value> a_values =
        a.size() == target_size ? decodeAFPValues(a) : std::vector<AFP::Value>{};
    const std::vector<AFP::Value> b_values =
        b.size() == target_size ? decodeAFPValues(b) : std::vector<AFP::Value>{};
    
    for (std::size_t i = 0; i < target_size; ++i) {
        const AFP::Value a_val =
            a.size() == 1 ? readAFPValue(a, 0, 0)
                          : a_values[i];
        const AFP::Value b_val =
            b.size() == 1 ? readAFPValue(b, 0, 0)
                          : b_values[i];
        output.push_back(a_val.add(b_val));
    }
    
    return buildTensorFromAFPValues(output, a.config_);
}

AFPEncodedTensor AFPArithmetic::broadcastMultiply(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b,
    std::size_t target_size)
{
    if ((a.size() != target_size && a.size() != 1) || (b.size() != target_size && b.size() != 1)) {
        throw std::invalid_argument("AFP broadcast multiply: invalid tensor sizes");
    }
    if (a.config_.block_size != b.config_.block_size) {
        throw std::invalid_argument("AFP block size mismatch");
    }
    
    std::vector<AFP::Value> output;
    output.reserve(target_size);
    
    /* Same bulk-decode pattern as broadcastAdd. */
    const std::vector<AFP::Value> a_values =
        a.size() == target_size ? decodeAFPValues(a) : std::vector<AFP::Value>{};
    const std::vector<AFP::Value> b_values =
        b.size() == target_size ? decodeAFPValues(b) : std::vector<AFP::Value>{};
    
    for (std::size_t i = 0; i < target_size; ++i) {
        const AFP::Value a_val =
            a.size() == 1 ? readAFPValue(a, 0, 0)
                          : a_values[i];
        const AFP::Value b_val =
            b.size() == 1 ? readAFPValue(b, 0, 0)
                          : b_values[i];
        output.push_back(a_val.multiply(b_val));
    }
    
    return buildTensorFromAFPValues(output, a.config_);
}