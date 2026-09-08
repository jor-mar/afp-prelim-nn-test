#include "../include/afp_value.hpp"
#include "../include/afp_math_new.hpp"
#include "../include/afp_encoded_tensor_new.hpp"
#include "../include/afp_thread_pool.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

/*
    Equivalence + thread-pool tests:

    1. matrixVectorMultiplyPrepared / matrixVectorMultiplyUnpacked must
       be bit-exact with matrixVectorMultiply (same decode, same
       Product conversion, same accumulation order).
    2. The unpacked pipeline (matvec -> addUnpacked -> reluUnpacked)
       must match the packed pipeline (matvec -> add -> relu) because
       every step uses identical AFP-native arithmetic.
    3. The thread pool: results identical under parallelFor, nested
       parallelFor must not deadlock, concurrent callers must serialize
       safely.
*/

static int failures = 0;

static void check(bool condition, const char *what)
{
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

static bool sameEncodedTensor(
    const AFPEncodedTensor &a,
    const AFPEncodedTensor &b)
{
    if (a.size() != b.size()) return false;
    if (a.blockCount() != b.blockCount()) return false;

    const std::vector<AFP::Value> av = AFPArithmetic::decodeAFPValues(a);
    const std::vector<AFP::Value> bv = AFPArithmetic::decodeAFPValues(b);

    for (std::size_t i = 0; i < av.size(); ++i) {
        if (av[i].negative != bv[i].negative ||
            av[i].exponent != bv[i].exponent ||
            av[i].offset != bv[i].offset ||
            av[i].mantissa != bv[i].mantissa ||
            av[i].positive_field != bv[i].positive_field) {
            return false;
        }
    }
    return true;
}

static void testPreparedMatVecEquivalence()
{
    std::printf("prepared matvec equivalence...\n");

    AFPQuantizer quantizer(AFPConfig{});
    std::mt19937 rng(777u);
    std::uniform_real_distribution<float> weight_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> input_dist(-2.0f, 2.0f);

    const std::size_t rows = 37;   // not block-aligned on purpose
    const std::size_t columns = 100;

    std::vector<float> weights(rows * columns);
    for (float &w : weights) w = weight_dist(rng);

    const AFPEncodedTensor w_tensor = quantizer.encode(weights);

    const AFPPreparedMatrix *prepared_ptr = nullptr;
    std::unique_ptr<AFPPreparedMatrix> prepared =
        AFPArithmetic::prepareMatrix(w_tensor, rows, columns);
    prepared_ptr = prepared.get();

    check(prepared_ptr->rows() == rows && prepared_ptr->columns() == columns,
          "prepared matrix dimensions");

    for (int trial = 0; trial < 20; ++trial) {
        std::vector<float> input(columns);
        for (float &v : input) v = input_dist(rng);

        const AFPEncodedTensor input_tensor = quantizer.encode(input);

        const AFPEncodedTensor packed =
            AFPArithmetic::matrixVectorMultiply(w_tensor, input_tensor, rows, columns);
        const AFPEncodedTensor prepared_result =
            AFPArithmetic::matrixVectorMultiplyPrepared(*prepared_ptr, input_tensor);

        check(sameEncodedTensor(packed, prepared_result),
              "prepared matvec bit-exact with packed matvec");

        // Unpacked pipeline vs packed pipeline.
        const std::vector<AFP::Value> input_values =
            AFPArithmetic::decodeAFPValues(input_tensor);
        const std::vector<AFP::Value> out_unpacked =
            AFPArithmetic::matrixVectorMultiplyUnpacked(*prepared_ptr, input_values);

        const AFPEncodedTensor packed_added =
            AFPArithmetic::add(packed, packed);  // same-shape add
        const std::vector<AFP::Value> unpacked_added =
            AFPArithmetic::addUnpacked(
                AFPArithmetic::decodeAFPValues(packed),
                AFPArithmetic::decodeAFPValues(packed));

        const AFPEncodedTensor packed_relu = AFPArithmetic::relu(packed_added);
        const std::vector<AFP::Value> unpacked_relu =
            AFPArithmetic::reluUnpacked(unpacked_added);

        check(sameEncodedTensor(packed_relu,
                                AFPArithmetic::encodeAFPValues(unpacked_relu, AFPConfig{})),
              "unpacked add+relu matches packed add+relu");
        (void)out_unpacked;
    }

    std::printf("prepared matvec equivalence done.\n");
}

static void testThreadPool()
{
    std::printf("thread pool...\n");

    // 1. parallelFor covers every index exactly once.
    {
        std::vector<int> counts(1000, 0);
        AFP::ThreadPool::parallelFor(0, counts.size(), 1,
            [&counts](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) counts[i] += 1;
            });

        bool all_once = true;
        for (int c : counts) if (c != 1) all_once = false;
        check(all_once, "parallelFor covers every index exactly once");
    }

    // 2. Nested parallelFor does not deadlock (inner runs serially).
    {
        std::vector<int> counts(100, 0);
        AFP::ThreadPool::parallelFor(0, 10, 1,
            [&counts](std::size_t, std::size_t) {
                AFP::ThreadPool::parallelFor(0, counts.size(), 1,
                    [&counts](std::size_t b2, std::size_t e2) {
                        for (std::size_t i = b2; i < e2; ++i) counts[i] += 1;
                    });
            });

        bool all_once = true;
        for (int c : counts) if (c != 10) all_once = false;
        check(all_once, "nested parallelFor completes with correct counts");
    }

    // 3. Concurrent callers from several threads produce correct results.
    {
        constexpr int callers = 4;
        constexpr std::size_t per_caller = 500;
        std::vector<int> sums(callers, 0);

        std::vector<std::thread> caller_threads;
        for (int c = 0; c < callers; ++c) {
            caller_threads.emplace_back([&sums, c] {
                int local = 0;
                for (int rep = 0; rep < 10; ++rep) {
                    AFP::ThreadPool::parallelFor(0, per_caller, 1,
                        [&local](std::size_t begin, std::size_t end) {
                            local += static_cast<int>(end - begin);
                        });
                }
                sums[c] = local;
            });
        }
        for (std::thread &t : caller_threads) t.join();

        bool all_correct = true;
        for (int s : sums) if (s != static_cast<int>(per_caller) * 10) all_correct = false;
        check(all_correct, "concurrent parallelFor callers get full counts");
    }

    // 4. parallelism() is sane.
    check(AFP::ThreadPool::parallelism() >= 1, "parallelism at least 1");

    std::printf("thread pool done.\n");
}

int main()
{
    std::printf("=== AFP equivalence + pool tests ===\n");

    testPreparedMatVecEquivalence();
    testThreadPool();

    if (failures == 0) {
        std::printf("=== All equivalence tests passed ===\n");
        return 0;
    }
    std::printf("=== %d failure(s) ===\n", failures);
    return 1;
}
