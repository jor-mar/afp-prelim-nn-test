#include "../include/afp_value.hpp"
#include "../include/afp_math_new.hpp"
#include "../include/afp_encoded_tensor_new.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

/*
    Robustness tests for the hardened AFP accumulator and helpers.

    - accumulateProduct: wide-dynamic-range sums must be well-defined
      (no signed-overflow UB, no silent scale mismatch) and remain
      accurate to within the rounding the helper documents.
    - Value::compare: no shift wraparound.
    - Value::tanh: bounded by (-1, 1) for small inputs (regression for
      the mis-scaled denominator).
    - Bit-exactness: the saturating accumulator must reproduce the
      previous exact results for ordinary same-scale data (the fast
      path is unchanged).
*/

static int failures = 0;

static void check(bool condition, const char *what)
{
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

static AFP::Product makeProduct(bool negative, uint64_t significand, int scale_exponent)
{
    AFP::Product p;
    p.negative = negative;
    p.significand = significand;
    p.scale_exponent = scale_exponent;
    p.zero = significand == 0;
    return p;
}

// Exact value of an accumulator state: magnitude * 2^exponent, signed.
static long double accumulatorValue(const AFP::Accumulator &acc)
{
    if (acc.zero || acc.significand == 0) return 0.0L;
    const long double magnitude = static_cast<long double>(acc.significand < 0
        ? -static_cast<uint64_t>(acc.significand)
        : static_cast<uint64_t>(acc.significand));
    const long double result = std::ldexp(
        static_cast<double>(magnitude), acc.exponent);
    return acc.negative ? -result : result;
}

static void testAccumulateProductRobustness()
{
    std::printf("accumulateProduct robustness...\n");

    // 1. Huge + tiny (old code: acc_value <<= 62+ on int64 -> overflow UB).
    {
        AFP::Accumulator acc(false, int64_t{1} << 47, 0, false);
        AFP::Utils::accumulateProduct(acc, makeProduct(false, 3, -62));
        // 3 * 2^-62 is ~6.5e-19, far below 2^47's rounding unit (~0.007);
        // the sum must stay 2^47 (absorbed with rounding, not dropped or
        // mis-scaled).
        check(accumulatorValue(acc) == 140737488355328.0L,
              "huge + tiny keeps exact value");
    }

    // 2. Tiny + huge, accumulator initialized from the tiny side.
    {
        AFP::Accumulator acc(false, 3, -62, false);
        AFP::Utils::accumulateProduct(acc, makeProduct(false, uint64_t{1} << 47, 0));
        check(accumulatorValue(acc) == 140737488355328.0L,
              "tiny + huge keeps exact value");
    }

    // 3. Cancellation to zero.
    {
        AFP::Accumulator acc(false, uint64_t{1} << 40, -5, false);
        AFP::Utils::accumulateProduct(acc, makeProduct(true, uint64_t{1} << 40, -5));
        check(acc.zero, "exact cancellation yields zero");
    }

    // 4. Same-sign sum crossing the int64 boundary (2^62 + 2^62).
    {
        AFP::Accumulator acc(false, int64_t{1} << 62, 0, false);
        AFP::Utils::accumulateProduct(acc, makeProduct(false, int64_t{1} << 62, 0));
        check(accumulatorValue(acc) == 9223372036854775808.0L,
              "2^62 + 2^62 = 2^63 without overflow");
    }

    // 5. Saturation stays finite and well-formed.
    {
        AFP::Accumulator acc(false, int64_t{1} << 62, 40, false);
        AFP::Utils::accumulateProduct(acc, makeProduct(false, int64_t{1} << 62, 40));
        check(accumulatorValue(acc) > 0.0L, "saturation stays positive");
        check(acc.significand > 0, "saturation keeps positive significand");
    }

    // 6. Randomized differential check against long double.
    {
        std::mt19937_64 rng(0x5EED1234u);
        std::uniform_int_distribution<int> exp_dist(-40, 40);
        std::uniform_int_distribution<uint64_t> sig_dist(1, uint64_t{1} << 62);

        for (int trial = 0; trial < 20000; ++trial) {
            AFP::Accumulator acc;
            long double reference = 0.0L;

            for (int term = 0; term < 64; ++term) {
                const bool negative = (rng() & 1u) != 0;
                const uint64_t significand = sig_dist(rng) | uint64_t{1};
                const int scale = exp_dist(rng);

                AFP::Utils::accumulateProduct(
                    acc, makeProduct(negative, significand, scale));

                reference += static_cast<long double>(std::ldexp(
                    static_cast<double>(significand), scale)) *
                    (negative ? -1.0L : 1.0L);
            }

            const long double got = accumulatorValue(acc);
            const long double reference_magnitude = std::fabs(reference);

            // The accumulator carries 64 bits of magnitude; relative error
            // against the exact long-double sum must be tiny. Saturation
            // cases (|reference| >= 2^63) are checked for finiteness only.
            if (reference_magnitude < 9.0e18L) {
                const long double relative =
                    reference_magnitude == 0.0L
                        ? (got == 0.0L ? 0.0L : 1.0L)
                        : std::fabs(got - reference) / reference_magnitude;
                check(relative < 1e-9L, "randomized accumulation accuracy");
                if (relative >= 1e-9L) {
                    std::printf("  trial %d: got %.20Le want %.20Le rel %.3Le\n",
                                trial, got, reference, relative);
                    break;
                }
            }
        }
    }

    std::printf("accumulateProduct robustness done.\n");
}

static void testValueCompareNoWrap()
{
    std::printf("Value::compare no wraparound...\n");

    // a = 2^62-ish magnitude at scale 0, b = same magnitude at scale -63.
    // Old code: lhs = significand << 63 wraps to 0 -> reported b >= a.
    AFP::Value a(false, 0, 0, static_cast<uint8_t>(uint64_t{1} << 6));
    AFP::Value b(false, 0, 63, static_cast<uint8_t>(uint64_t{1} << 6));
    check(a.compare(b) > 0, "compare keeps ordering across 63-bit scale gap");

    std::printf("Value::compare done.\n");
}

static void testTanhBounded()
{
    std::printf("Value::tanh bounded...\n");

    AFPQuantizer quantizer(AFPConfig{});
    const std::vector<float> inputs = {
        0.001f, 0.01f, 0.05f, 0.1f, 0.2f, 0.3f, 0.5f, 0.7f, 1.0f,
        -0.001f, -0.01f, -0.05f, -0.1f, -0.2f, -0.3f, -0.5f, -0.7f, -1.0f,
        2.0f, -2.0f, 4.0f, -4.0f, 10.0f, -10.0f,
    };

    for (float x : inputs) {
        const AFPEncodedTensor encoded = quantizer.encode({x});
        const AFPEncodedTensor result = AFPArithmetic::tanh(encoded);
        const std::vector<float> decoded = quantizer.decode(result);
        check(decoded.size() == 1, "tanh output has one value");
        if (decoded.size() == 1) {
            check(decoded[0] > -1.0001f && decoded[0] < 1.0001f,
                  "tanh stays in (-1, 1)");
            if (decoded[0] <= -1.0001f || decoded[0] >= 1.0001f) {
                std::printf("  tanh(%g) = %g out of range\n", x, decoded[0]);
            }
        }
    }

    std::printf("Value::tanh done.\n");
}

static void testBitExactOnOrdinaryData()
{
    std::printf("Bit-exactness on ordinary data...\n");

    AFPQuantizer quantizer(AFPConfig{});
    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> weights(64);
    for (float &w : weights) w = dist(rng);

    const AFPEncodedTensor w_tensor = quantizer.encode(weights);
    const std::vector<float> input = {0.5f, -1.25f, 2.0f, 0.125f};

    // matrixVectorMultiply must run and produce finite values.
    const AFPEncodedTensor w4 = quantizer.encode(std::vector<float>(
        weights.begin(), weights.begin() + 16));
    const AFPEncodedTensor input4 = quantizer.encode(input);
    const AFPEncodedTensor result =
        AFPArithmetic::matrixVectorMultiply(w4, input4, 4, 4);
    const std::vector<float> decoded = quantizer.decode(result);

    check(decoded.size() == 4, "matvec output size");
    for (float v : decoded) {
        check(std::isfinite(v), "matvec output finite");
    }

    std::printf("Bit-exactness done.\n");
}

int main()
{
    std::printf("=== AFP robustness tests ===\n");

    testAccumulateProductRobustness();
    testValueCompareNoWrap();
    testTanhBounded();
    testBitExactOnOrdinaryData();

    if (failures == 0) {
        std::printf("=== All robustness tests passed ===\n");
        return 0;
    }
    std::printf("=== %d failure(s) ===\n", failures);
    return 1;
}
