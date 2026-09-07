/*
    Regression test for the positive-field mantissa width at offset == 7.

    When a half block of 8 values is all positive, the sign bit is reused
    as a 6th mantissa bit. For offset == 7 (no implicit leading 1) such a
    value can store a raw mantissa below 32; the AFP-native arithmetic path
    must still treat it as a 6-bit field. It previously inferred the width
    from the mantissa magnitude, scaled those values 2x too large, and
    disagreed with the codec.

    The test encodes such a block, pushes it through AFPArithmetic::absolute
    (an identity for positive values, forcing decode -> toProduct ->
    re-encode), and requires the decoded result to match the codec's own
    decode of the original tensor within 1% relative error per value.
*/

#include "../include/afp_encoded_tensor_new.hpp"
#include "../include/afp_math_new.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

int main()
{
    AFPConfig config;
    AFPQuantizer quantizer(config);

    // Block of 16, both halves all positive. First half ~1.0 so the shared
    // exponent is 0; second half tiny values that quantize to offset == 7
    // with raw mantissas straddling 32 (8, 16, 25, 33, 41, 49, 57, 63).
    std::vector<float> data = {
        1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f,
        0.001f, 0.002f, 0.003f, 0.004f, 0.005f, 0.006f, 0.007f, 0.008f,
    };

    AFPEncodedTensor encoded = quantizer.encode(data);
    const std::vector<float> direct = quantizer.decode(encoded);

    AFPEncodedTensor identity = AFPArithmetic::absolute(encoded);
    const std::vector<float> roundtrip = quantizer.decode(identity);

    double worst_relative = 0.0;
    bool failed = false;

    for (std::size_t i = 0; i < data.size(); ++i) {
        const double reference = direct[i];
        const double error = std::abs(roundtrip[i] - reference);
        const double relative = (reference != 0.0) ? error / std::abs(reference) : error;

        if (relative > worst_relative) worst_relative = relative;
        if (relative > 0.01) failed = true;

        std::printf("[%2zu] in=%9.3g  codec=%9.4g  arithmetic=%9.4g  rel_err=%.4f\n",
                    i, data[i], direct[i], roundtrip[i], relative);
    }

    std::printf("worst relative error: %g -> %s\n",
                worst_relative, failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
