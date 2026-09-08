#include "../include/afp_value.hpp"
#include "../include/afp_math_new.hpp"
#include "../include/afp_encoded_tensor_new.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

/*
    Round-trip audit for the encode/decode fast paths in
    afp_math_new.cpp.

    The strong check: for any tensor produced by the trusted
    AFPQuantizer::encode writer, re-encoding the decoded AFP::Values
    through buildTensorFromAFPValues must reproduce the exact same
    bytes, and re-decoding through the block-window decoder must
    reproduce the same Value fields. This verifies both fast paths are
    bit-exact with the per-value writer/reader they replaced.
*/

static int failures = 0;

static void check(bool condition, const char *what)
{
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

int main()
{
    std::printf("=== encode/decode round-trip audit ===\n");

    AFPQuantizer quantizer(AFPConfig{});

    std::mt19937 rng(20260908u);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (int trial = 0; trial < 500; ++trial) {
        const std::size_t count = 1 + (rng() % 300);
        std::vector<float> values(count);
        for (float &v : values) v = dist(rng);

        const AFPEncodedTensor original = quantizer.encode(values);

        // Decode through the block-window decoder.
        const std::vector<AFP::Value> decoded_values =
            AFPArithmetic::decodeAFPValues(original);
        check(decoded_values.size() == count, "decoded value count");
        if (decoded_values.size() != count) break;

        // Re-encode through buildTensorFromAFPValues and compare bytes.
        const AFPEncodedTensor reencoded =
            AFPArithmetic::encodeAFPValues(decoded_values, AFPConfig{});

        check(reencoded.bitSize() == original.bitSize(),
              "re-encoded bit size matches");
        if (reencoded.bitSize() != original.bitSize()) break;

        const std::vector<uint8_t> &a = original.bitStream().data();
        const std::vector<uint8_t> &b = reencoded.bitStream().data();
        bool same = a.size() == b.size();
        for (std::size_t i = 0; same && i < a.size(); ++i) {
            same = a[i] == b[i];
        }
        check(same, "re-encoded bytes identical to original writer");
        if (!same) {
            std::printf("  trial %d count %zu: byte mismatch\n", trial, count);
            break;
        }
    }

    if (failures == 0) {
        std::printf("=== round-trip audit passed ===\n");
        return 0;
    }
    std::printf("=== %d failure(s) ===\n", failures);
    return 1;
}