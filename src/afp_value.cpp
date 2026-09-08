#include "../include/afp_value.hpp"
#include <stdexcept>
#include <algorithm>
#include <limits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace AFP {

// ============================================================================
// Utility Functions
// ============================================================================

namespace Utils {

/*
    Floor log2 via count-leading-zeros. Hardware CLZ/BSR instruction on
    GCC/Clang/MSVC; portable bit loop otherwise. Bit-exact with the
    previous shift-loop implementation for every input including 0.
*/

int integerLog2(uint64_t value) {
    if (value == 0) return -1;

#if defined(__GNUC__) || defined(__clang__)
    return 63 - static_cast<int>(__builtin_clzll(value));
#elif defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanReverse64(&index, value);
    return static_cast<int>(index);
#else
    int result = -1;
    while (value != 0) {
        value >>= 1;
        ++result;
    }
    return result;
#endif
}

uint64_t shiftRightRounded(uint64_t value, int shift) {
    if (shift <= 0) return value;
    if (shift >= 64) return 0;
    
    const uint64_t truncated = value >> shift;
    const uint64_t remainder = value & maskBits(shift);
    const uint64_t halfway = uint64_t{1} << (shift - 1);
    
    if (remainder > halfway) return truncated + 1;
    if (remainder == halfway && (truncated & 1U)) return truncated + 1;
    
    return truncated;
}

uint64_t maskBits(int bit_count) {
    if (bit_count <= 0) return 0;
    if (bit_count >= 64) return std::numeric_limits<uint64_t>::max();
    return (uint64_t{1} << bit_count) - 1;
}

int decodeSharedExponent(uint8_t encoded) {
    if (encoded == 0) return -126;
    return static_cast<int>(encoded) - 127;
}

uint8_t encodeSharedExponent(int exponent) {
    if (exponent < -126) exponent = -126;
    if (exponent > 127) exponent = 127;
    return static_cast<uint8_t>(exponent + 127);
}

void normalizeAccumulator(Accumulator& acc) {
    if (acc.zero || acc.significand == 0) {
        acc.zero = true;
        acc.negative = false;
        acc.significand = 0;
        acc.exponent = 0;
        return;
    }
    
    const uint64_t magnitude = static_cast<uint64_t>(acc.significand);
    constexpr int accumulator_bits = 48;
    const int highest_bit = integerLog2(magnitude);
    
    if (highest_bit >= accumulator_bits) {
        const int shift = highest_bit - accumulator_bits + 1;
        acc.significand = static_cast<int64_t>(shiftRightRounded(magnitude, shift));
        acc.exponent += shift;
    }
}

/*
    Accumulate one product into a running accumulator.

    Well-defined for every input (AFP-native, integer only):

      - Magnitudes are handled as uint64 with the sign carried
        separately, so no signed shift or addition is ever evaluated
        (the previous implementation shifted and added int64 values
        and could overflow / wrap).

      - Fast path (bit-exact with the previous implementation): align
        both terms at the smaller scale and shift each magnitude up.
        This path is taken only when every shifted magnitude and the
        magnitude of the sum fit in 64 bits.

      - Fallback (previously undefined or silently mis-scaled): when a
        term would need a >= 64-bit up-shift or the same-sign sum
        overflows, align instead at the larger term's top bit. The
        larger magnitude is shifted down with rounding and the smaller
        term always fits the remaining envelope, so the result keeps
        the correct scale; the sum saturates at the largest
        representable magnitude only if even that is exceeded.
*/

void accumulateProduct(Accumulator& acc, const Product& product) {
    if (product.zero || product.significand == 0) {
        return;
    }

    const uint64_t prod_mag = product.significand;

    if (acc.zero || acc.significand == 0) {
        acc.negative = product.negative;
        acc.significand = static_cast<int64_t>(prod_mag);
        acc.exponent = product.scale_exponent;
        acc.zero = false;
        normalizeAccumulator(acc);
        return;
    }

    const uint64_t acc_mag = static_cast<uint64_t>(acc.significand);
    const bool same_sign = (acc.negative == product.negative);

    /*
        Fast path: exact alignment at the smaller scale.
    */

    const int common_exponent =
        std::min(acc.exponent, product.scale_exponent);
    const int acc_up = acc.exponent - common_exponent;
    const int prod_up = product.scale_exponent - common_exponent;

    const int acc_top = static_cast<int>(integerLog2(acc_mag));
    const int prod_top = static_cast<int>(integerLog2(prod_mag));

    if (acc_up < 64 && prod_up < 64 &&
        acc_top + acc_up <= 63 && prod_top + prod_up <= 63) {

        const uint64_t a = acc_mag << acc_up;
        const uint64_t b = prod_mag << prod_up;

        uint64_t magnitude;
        bool negative;

        if (same_sign) {
            magnitude = a + b;
            if (magnitude < a) {
                /* Magnitude sum exceeds 64 bits: saturate. */
                acc.negative = acc.negative;
                acc.significand = static_cast<int64_t>(~uint64_t{0} >> 1);
                acc.exponent = common_exponent + 63;
                acc.zero = false;
                return;
            }
            if (magnitude >= uint64_t{1} << 63) {
                /*
                    Keeps the magnitude int64-representable: round one
                    bit down and carry the extra factor into the scale.
                */
                magnitude = shiftRightRounded(magnitude, 1);
                acc.negative = acc.negative;
                acc.significand = static_cast<int64_t>(magnitude);
                acc.exponent = common_exponent + 1;
                acc.zero = false;
                normalizeAccumulator(acc);
                return;
            }
            negative = acc.negative;
        } else {
            if (a >= b) {
                magnitude = a - b;
                negative = acc.negative;
            } else {
                magnitude = b - a;
                negative = product.negative;
            }
        }

        if (magnitude == 0) {
            acc.zero = true;
            acc.negative = false;
            acc.significand = 0;
            acc.exponent = 0;
            return;
        }

        acc.negative = negative;
        acc.significand = static_cast<int64_t>(magnitude);
        acc.exponent = common_exponent;
        acc.zero = false;
        normalizeAccumulator(acc);
        return;
    }

    /*
        Fallback: align both terms at the exponent that puts the top
        bit of the larger term at bit 63. For each term the move from
        its own exponent to the alignment exponent satisfies

            move = exponent - align = 63 - (top + exponent - align)

        relative to its scale top, so an up-shift is at most 63 bits
        and the shifted magnitude's top bit lands at or below bit 63:
        no shift can overflow. A negative move (only possible for the
        smaller term) shifts down with rounding, absorbing terms that
        are negligible at the alignment scale.
    */

    const int acc_scale_top = acc_top + acc.exponent;
    const int prod_scale_top = prod_top + product.scale_exponent;

    int align_exponent =
        std::max(acc_scale_top, prod_scale_top) - 63;

    const int acc_move = acc.exponent - align_exponent;
    const int prod_move = product.scale_exponent - align_exponent;

    uint64_t a = acc_mag;
    uint64_t b = prod_mag;

    if (acc_move > 0) a = a << acc_move;
    else if (acc_move < 0) a = shiftRightRounded(a, -acc_move);

    if (prod_move > 0) b = b << prod_move;
    else if (prod_move < 0) b = shiftRightRounded(b, -prod_move);

    uint64_t magnitude;
    bool negative;

    if (same_sign) {
        magnitude = a + b;
        if (magnitude < a) {
            /*
                The larger term's top bit sits at bit 63, so the sum
                can exceed the 64-bit envelope: saturate (the final
                round-and-rescale below keeps the result well-formed).
            */
            magnitude = ~uint64_t{0};
        }
        negative = acc.negative;
    } else {
        if (a >= b) {
            magnitude = a - b;
            negative = acc.negative;
        } else {
            magnitude = b - a;
            negative = product.negative;
        }
    }

    if (magnitude == 0) {
        acc.zero = true;
        acc.negative = false;
        acc.significand = 0;
        acc.exponent = 0;
        return;
    }

    if (magnitude >= uint64_t{1} << 63) {
        /*
            Keep the magnitude int64-representable: round one bit down
            and carry the extra factor of two into the scale.
        */
        magnitude = shiftRightRounded(magnitude, 1);
        ++align_exponent;
    }

    acc.negative = negative;
    acc.significand = static_cast<int64_t>(magnitude);
    acc.exponent = align_exponent;
    acc.zero = false;
    normalizeAccumulator(acc);
}

} // namespace Utils

// ============================================================================
// Precision Control
// ============================================================================

namespace {

int g_division_precision_bits = 24;

/*
    Six-term Taylor expansion of e^g, valid for |g| <= ~0.7 where the
    truncation error (g^7/7!) is below 0.1%.

    The coefficients are the closest AFP encodings with mantissas up to
    ~8 bits (mantissas beyond 63 are fine for internal constants: the
    Value struct stores an integer mantissa and toProduct() scales it by
    exponent - offset - mantissaBits()):

        1/6   = 171 * 2^-10   (+0.20%)
        1/24  = 171 * 2^-12   (+0.20%)
        1/120 = 137 * 2^-14   (+0.34%)
        1/720 =  91 * 2^-16   (-0.02%)

    Each coefficient's error is weighted by g^k/k! in the sum, so it
    stays below 0.05% in absolute terms.
*/
Value expSeries(const Value& g) {
    const Value one = Value::oneValue();

    Value half;
    half.negative = false;
    half.exponent = -1;
    half.offset = 0;
    half.mantissa = static_cast<uint8_t>(32);

    Value one_sixth;
    one_sixth.negative = false;
    one_sixth.exponent = -5;
    one_sixth.offset = 0;
    one_sixth.mantissa = static_cast<uint8_t>(171);

    Value one_24th;
    one_24th.negative = false;
    one_24th.exponent = -7;
    one_24th.offset = 0;
    one_24th.mantissa = static_cast<uint8_t>(171);

    Value one_120th;
    one_120th.negative = false;
    one_120th.exponent = -9;
    one_120th.offset = 0;
    one_120th.mantissa = static_cast<uint8_t>(137);

    Value one_720th;
    one_720th.negative = false;
    one_720th.exponent = -11;
    one_720th.offset = 0;
    one_720th.mantissa = static_cast<uint8_t>(91);

    const Value g2 = g.multiply(g);
    const Value g3 = g2.multiply(g);
    const Value g4 = g3.multiply(g);
    const Value g5 = g4.multiply(g);
    const Value g6 = g5.multiply(g);

    Value result = one.add(g);
    result = result.add(g2.multiply(half));
    result = result.add(g3.multiply(one_sixth));
    result = result.add(g4.multiply(one_24th));
    result = result.add(g5.multiply(one_120th));
    result = result.add(g6.multiply(one_720th));

    return result;
}

} // namespace

void Precision::setDivisionPrecisionBits(int bits) {
    if (bits < 0 || bits > 63) {
        throw std::invalid_argument("AFP division precision must be in [0, 63] bits");
    }
    g_division_precision_bits = bits;
}

int Precision::divisionPrecisionBits() {
    return g_division_precision_bits;
}

// ============================================================================
// Value Implementation
// ============================================================================

bool Value::isZero() const {
    return offset == 7 && mantissa == 0;
}

bool Value::isNegative() const {
    return negative && !isZero();
}

bool Value::isPositive() const {
    return !negative && !isZero();
}

int Value::compare(const Value& other) const {
    const bool a_zero = isZero();
    const bool b_zero = other.isZero();
    
    if (a_zero && b_zero) return 0;
    if (a_zero) return other.negative ? 1 : -1;
    if (b_zero) return negative ? -1 : 1;
    
    if (negative != other.negative) return negative ? -1 : 1;
    
    const Product pa = toProduct();
    const Product pb = other.toProduct();
    
    const int a_top = pa.scale_exponent + Utils::integerLog2(pa.significand);
    const int b_top = pb.scale_exponent + Utils::integerLog2(pb.significand);
    
    if (a_top < b_top) return -1;
    if (a_top > b_top) return 1;
    
    if (pa.scale_exponent == pb.scale_exponent) {
        if (pa.significand < pb.significand) return -1;
        if (pa.significand > pb.significand) return 1;
        return 0;
    }
    
    /*
        Compare at a common scale. An up-shift that would push a
        magnitude out of its 64-bit representation means that operand
        is strictly larger (the previous code shifted anyway and
        wrapped, mis-ordering such pairs).
    */
    const int shift = pa.scale_exponent - pb.scale_exponent;
    if (shift > 0) {
        if (shift >= 64 || pa.significand > (std::numeric_limits<uint64_t>::max() >> shift)) {
            return 1;
        }
        const uint64_t lhs = pa.significand << shift;
        return lhs < pb.significand ? -1 : (lhs > pb.significand ? 1 : 0);
    } else {
        const int reverse_shift = -shift;
        if (reverse_shift >= 64 || pb.significand > (std::numeric_limits<uint64_t>::max() >> reverse_shift)) {
            return -1;
        }
        const uint64_t rhs = pb.significand << reverse_shift;
        return pa.significand < rhs ? -1 : (pa.significand > rhs ? 1 : 0);
    }
}

Value Value::negate() const {
    if (isZero()) return *this;
    Value result = *this;
    result.negative = !result.negative;
    return result;
}

Value Value::absolute() const {
    Value result = *this;
    result.negative = false;
    return result;
}

Value Value::relu() const {
    if (isNegative()) return zero();
    return *this;
}

Value Value::scalePowerOfTwo(int power) const {
    if (isZero()) return *this;
    Value result = *this;
    int new_exponent = static_cast<int>(exponent) + power;
    if (new_exponent < -126) new_exponent = -126;
    if (new_exponent > 127) new_exponent = 127;
    result.exponent = static_cast<int8_t>(new_exponent);
    return result;
}

// ============================================================================
// Product Conversion
// ============================================================================

Product Value::toProduct() const {
    Product result;
    if (isZero()) return result;
    
    result.negative = negative;
    result.significand = mantissa;
    const int mant_bits = mantissaBits();
    result.scale_exponent = static_cast<int>(exponent) - static_cast<int>(offset) - mant_bits;
    result.zero = false;
    return result;
}

// ============================================================================
// AFP-Native Arithmetic Operations
//
// Every operation below runs on the integer significand/scale-exponent
// form (Product/Accumulator) using integer arithmetic only. None of them
// converts to float or double; float conversion exists solely in the
// codec boundary (AFPQuantizer::encode / AFPQuantizer::decode).
// ============================================================================

Value Value::add(const Value& other) const {
    // Convert to products for normalization
    const Product prod_a = toProduct();
    const Product prod_b = other.toProduct();
    
    //
    // Accumulate through the shared saturating helper: exponent
    // alignment and the sum itself are well-defined for every input
    // (the previous inline int64 shifts/addition could overflow).
    //
    
    Accumulator acc;
    
    if (!prod_a.zero && prod_a.significand != 0) {
        acc.negative = prod_a.negative;
        acc.significand = static_cast<int64_t>(prod_a.significand);
        acc.exponent = prod_a.scale_exponent;
        acc.zero = false;
        Utils::normalizeAccumulator(acc);
    }
    
    Utils::accumulateProduct(acc, prod_b);
    
    if (acc.zero) return zero();
    
    const int new_exponent = acc.exponent + Utils::integerLog2(static_cast<uint64_t>(acc.significand));
    return fromAccumulator(acc, new_exponent, false);
}

Value Value::subtract(const Value& other) const {
    Value negated_other = other;
    if (!other.isZero()) negated_other.negative = !negated_other.negative;
    return add(negated_other);
}

Value Value::multiply(const Value& other) const {
    if (isZero() || other.isZero()) return zero();
    
    const Product prod_a = toProduct();
    const Product prod_b = other.toProduct();
    
    Product result;
    result.negative = prod_a.negative ^ prod_b.negative;
    result.significand = prod_a.significand * prod_b.significand;
    result.scale_exponent = prod_a.scale_exponent + prod_b.scale_exponent;
    result.zero = result.significand == 0;
    
    if (result.zero) return zero();
    
    Accumulator acc;
    acc.negative = result.negative;
    acc.significand = static_cast<int64_t>(result.significand);
    acc.exponent = result.scale_exponent;
    acc.zero = false;
    
    const int new_exponent = acc.exponent + Utils::integerLog2(static_cast<uint64_t>(acc.significand));
    return fromAccumulator(acc, new_exponent, false);
}

Value Value::divide(const Value& other) const {
    if (isZero()) return zero();
    if (other.isZero()) throw std::invalid_argument("AFP division by zero");
    
    const Product numerator = toProduct();
    const Product denominator = other.toProduct();
    
    if (numerator.zero) return zero();
    
    const int division_precision_bits = Precision::divisionPrecisionBits();
    
    if (numerator.significand > (std::numeric_limits<uint64_t>::max() >> division_precision_bits)) {
        throw std::overflow_error("AFP division numerator overflow");
    }
    
    const uint64_t scaled_numerator = numerator.significand << division_precision_bits;
    const uint64_t quotient = scaled_numerator / denominator.significand;
    const uint64_t remainder = scaled_numerator % denominator.significand;
    
    uint64_t rounded_quotient = quotient;
    if (remainder >= denominator.significand - remainder) {
        if (rounded_quotient == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("AFP division quotient overflow");
        }
        ++rounded_quotient;
    }
    
    if (rounded_quotient == 0) return zero();
    
    Product result;
    result.negative = numerator.negative ^ denominator.negative;
    result.significand = rounded_quotient;
    result.scale_exponent = numerator.scale_exponent - denominator.scale_exponent - division_precision_bits;
    result.zero = false;
    
    Accumulator acc;
    acc.negative = result.negative;
    acc.significand = static_cast<int64_t>(result.significand);
    acc.exponent = result.scale_exponent;
    acc.zero = false;
    
    const int new_exponent = acc.exponent + Utils::integerLog2(static_cast<uint64_t>(acc.significand));
    return fromAccumulator(acc, new_exponent, false);
}

// ============================================================================
// Specialized Operations
// ============================================================================

Value Value::reciprocal() const {
    if (isZero()) throw std::invalid_argument("AFP reciprocal: division by zero");
    
    const Product p = toProduct();
    const int top = p.scale_exponent + Utils::integerLog2(p.significand);
    
    Value estimate;
    estimate.negative = negative;
    estimate.exponent = static_cast<int8_t>(-top);
    estimate.offset = 0;
    estimate.mantissa = static_cast<uint8_t>(32);
    
    const Value two = twoValue();
    
    // Newton-Raphson iteration
    for (int iteration = 0; iteration < 3; ++iteration) {
        const Value xy = multiply(estimate);
        const Value correction = two.subtract(xy);
        estimate = estimate.multiply(correction);
        if (estimate.isZero()) break;
    }
    
    return estimate;
}

Value Value::sqrt() const {
    if (isZero()) return zero();
    if (negative) throw std::invalid_argument("AFP sqrt: negative input");
    
    const Product p = toProduct();
    const int top = p.scale_exponent + Utils::integerLog2(p.significand);
    const int initial_exponent = top / 2;
    
    Value estimate;
    estimate.negative = false;
    estimate.exponent = static_cast<int8_t>(initial_exponent);
    estimate.offset = 0;
    estimate.mantissa = static_cast<uint8_t>(32);
    
    const Value half = halfValue();
    
    // Newton-Raphson iteration
    for (int iteration = 0; iteration < 4; ++iteration) {
        const Value quotient = divide(estimate);
        const Value sum = estimate.add(quotient);
        estimate = sum.multiply(half);
    }
    
    return estimate;
}

Value Value::reciprocalSqrt() const {
    const Value root = sqrt();
    return root.reciprocal();
}

Value Value::exp() const {
    if (isZero()) return oneValue();

    /*
        e^x = 2^m * e^g with m = round(x * log2(e)) and
        g = x - m * ln(2) in [-ln2/2, ln2/2].

        m and g are computed exactly in the integer domain: log2(e) and
        ln(2) enter as 24-bit fixed-point constants, so the only
        rounding is the final quantization of g to the mantissa grid
        (the 6-term Taylor series below has < 0.1% truncation error for
        |g| <= 0.7). The 2^m factor is applied exactly through the
        exponent field, so no squaring amplifies rounding error.

        This replaces the old direct Taylor evaluation, which diverged
        catastrophically outside [-2, 2] (e^-5 returned ~23, e^10 was
        ~10x low), silently breaking softmax, whose shifted arguments
        are always <= 0.
    */

    const Product xp = toProduct();
    const int64_t x_sign = xp.negative ? -1 : 1;

    // log2(e) ~= 1.4426950408889634, ln(2) ~= 0.6931471805599453
    constexpr int64_t kLog2e24 = 24204406;   // round(log2(e) * 2^24)
    constexpr int64_t kLn224   = 11629080;   // round(ln(2) * 2^24)

    /*
        m = round(x * log2(e)): y = x_sign * significand * 2^scale,
        scaled by 2^24 so the constant is exact to 24 bits.
    */
    const int64_t y_scaled = x_sign * static_cast<int64_t>(xp.significand) * kLog2e24;
    const int32_t y_shift = xp.scale_exponent - 24;

    int64_t m = 0;
    if (y_shift >= 0) {
        m = (y_shift < 63) ? (y_scaled << y_shift)
                           : (y_scaled > 0 ? INT64_MAX : INT64_MIN);
    } else {
        const int32_t shift = -y_shift;
        if (shift >= 64) {
            m = 0;
        } else {
            const int64_t half = y_scaled >= 0
                ? (int64_t{1} << (shift - 1))
                : -(int64_t{1} << (shift - 1));
            m = (y_scaled + half) / (int64_t{1} << shift);
        }
    }

    if (m > 127) {
        /* Saturate: e^x exceeds the format's exponent range. */
        Value saturated;
        saturated.negative = false;
        saturated.exponent = 127;
        saturated.offset = 0;
        saturated.mantissa = static_cast<uint8_t>(63);
        return saturated;
    }
    if (m < -127) {
        /* Underflow: e^x is below the format's exponent range. */
        return zero();
    }

    /*
        g = x - m * ln(2), computed exactly in the integer domain:
        g * 2^24 = x_sign * significand * 2^(scale + 24) - m * ln2_24.
    */
    Value g;
    if (m == 0) {
        g = *this;
    } else {
        const int32_t g_shift = xp.scale_exponent + 24;
        const int64_t x_scaled =
            x_sign * static_cast<int64_t>(xp.significand) * (int64_t{1} << g_shift);
        const int64_t B = x_scaled - m * kLn224;

        Product gp;
        gp.negative = B < 0;
        gp.significand = static_cast<uint64_t>(B < 0 ? -B : B);
        gp.scale_exponent = -24;
        gp.zero = false;

        const int top = gp.scale_exponent +
            Utils::integerLog2(gp.significand);
        g = Value::fromProduct(gp, top, false);
    }

    Value result = expSeries(g);
    return result.scalePowerOfTwo(static_cast<int>(m));
}

Value Value::tanh() const {
    if (isZero()) return *this;
    
    const Product product = toProduct();
    const int highest = product.scale_exponent + Utils::integerLog2(product.significand);
    
    /*
        Check if value is large enough to saturate. The rational
        approximation below reaches 1 exactly at |x| = 3 and exceeds 1
        beyond it, so the saturation threshold is |x| >= 3, i.e.
        significand >= 3 * 2^(-scale_exponent). (The previous test
        shifted the wrong way and only saturated for |x| >= 6, letting
        inputs in [3, 6) produce tanh values above 1.) When top == 1
        the scale is <= 0, so -scale_exponent fits the shift safely.
    */
    bool large = false;
    if (highest > 1) {
        large = true;
    } else if (highest == 1) {
        const int shift = -product.scale_exponent;
        if (shift >= 0 && shift < 64) {
            large = product.significand >= (uint64_t{3} << shift);
        }
    }
    
    if (large) {
        Value result;
        result.negative = negative;
        result.exponent = 0;
        result.offset = 0;
        result.mantissa = uint64_t{1} << 5;
        return result;
    }
    
    // tanh(x) = (27x + x^3) / (27 + 9x^2) using rational approximation
    const Value square = multiply(*this);
    
    // Numerator: 27x + x^3
    /*
        Constant 27 = 54 * 2^(4 - 0 - 5). (The previous encoding used
        mantissa 34, which evaluates to 17 and skewed every tanh.)
    */
    Value twenty_seven;
    twenty_seven.negative = false;
    twenty_seven.exponent = 4;
    twenty_seven.offset = 0;
    twenty_seven.mantissa = static_cast<uint8_t>(54);
    
    const Value numerator_part = twenty_seven.add(square);
    const Value numerator = multiply(numerator_part);
    
    // Denominator: 27 + 9x^2
    Product nine_square = square.toProduct();
    nine_square.significand *= 9;
    
    Accumulator denom_acc;
    denom_acc.negative = false;
    denom_acc.significand = 27;
    denom_acc.exponent = 0;
    denom_acc.zero = false;
    
    /*
        Sum 27 + 9x^2 through the saturating accumulator helper: the
        exponent alignment is exact and well-defined for every input.
        (The previous inline alignment shifted the denominator up when
        x was small but then interpreted the sum at the denominator's
        old scale, inflating it by 2^-scale_exponent and producing
        tanh values outside (-1, 1).)
    */
    Utils::accumulateProduct(denom_acc, nine_square);
    
    const int denominator_exponent =
        denom_acc.exponent +
        Utils::integerLog2(static_cast<uint64_t>(denom_acc.significand));
    
    const Value denominator = fromAccumulator(denom_acc, denominator_exponent, false);
    
    Value result = numerator.divide(denominator);
    
    /*
        Clamp the magnitude to <= 1: quantization of numerator and
        denominator can push the ratio marginally past 1 for |x| just
        under 3. Compare magnitudes (divide() always returns a
        sign-bit value, so absolute() is exact) — comparing the raw
        result against one() misses negative overshoots.
    */
    if (result.absolute().compare(one()) > 0) {
        result = one();
        result.negative = negative;
    }
    return result;
}

// ============================================================================
// Conversion from Product/Accumulator
// ============================================================================

Value Value::fromProduct(const Product& prod, int shared_exponent, bool positive_field) {
    Accumulator acc;
    acc.negative = prod.negative;
    acc.significand = static_cast<int64_t>(prod.significand);
    acc.exponent = prod.scale_exponent;
    acc.zero = prod.zero;
    return fromAccumulator(acc, shared_exponent, positive_field);
}

Value Value::fromAccumulator(const Accumulator& acc, int shared_exponent, bool positive_field) {
    Value result;
    result.exponent = static_cast<int8_t>(shared_exponent);
    result.negative = acc.negative;
    result.positive_field = positive_field;
    
    const int mantissa_bits = positive_field ? 6 : 5;
    
    if (acc.zero || acc.significand == 0) {
        result.negative = false;
        result.offset = 7;
        result.mantissa = 0;
        return result;
    }
    
    const int highest_bit = Utils::integerLog2(static_cast<uint64_t>(acc.significand));
    const int value_exponent = acc.exponent + highest_bit;
    
    int offset = shared_exponent - value_exponent;
    if (offset < 0) offset = 0;
    if (offset > 7) offset = 7;
    
    result.offset = static_cast<uint8_t>(offset);
    
    const int target_exponent = shared_exponent - offset - mantissa_bits;
    const int shift = target_exponent - acc.exponent;
    
    uint64_t scaled;
    if (shift >= 0) {
        if (shift >= 64) scaled = 0;
        else scaled = Utils::shiftRightRounded(static_cast<uint64_t>(acc.significand), shift);
    } else {
        const int left_shift = -shift;
        if (left_shift >= 63) scaled = std::numeric_limits<uint64_t>::max();
        else scaled = static_cast<uint64_t>(acc.significand) << left_shift;
    }
    
    if (offset < 7) {
        const uint64_t implicit = uint64_t{1} << mantissa_bits;
        const uint64_t maximum = (implicit << 1) - 1;
        if (scaled < implicit) scaled = implicit;
        if (scaled > maximum) scaled = maximum;
        result.mantissa = static_cast<uint8_t>(scaled);
    } else {
        const uint64_t maximum = Utils::maskBits(mantissa_bits);
        if (scaled > maximum) scaled = maximum;
        result.mantissa = static_cast<uint8_t>(scaled);
    }
    
    if (result.isZero()) result.negative = false;
    
    return result;
}

} // namespace AFP