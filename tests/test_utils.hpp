#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../src/bitstream.cpp"
#include "../src/encoded_tensor.cpp"

namespace test_utils
{
    int &failureCount()
    {
        static int failures = 0;
        return failures;
    }

    void expect(
        bool condition,
        const std::string &test_name
    )
    {
        if (condition)
        {
            std::cout << "[PASS] " << test_name << '\n';
        }
        else
        {
            std::cerr << "[FAIL] " << test_name << '\n';
            ++failureCount();
        }
    }

    void expectNear(
        double actual,
        double expected,
        double tolerance,
        const std::string &test_name
    )
    {
        const bool passed = std::abs(actual - expected) <= tolerance;

        expect(passed, test_name);

        if (!passed)
        {
            std::cerr << "\tExpected: " << expected << '\n';
            std::cerr << "\tActual: " << actual << '\n';
            std::cerr << "\tTolerance: " << tolerance << '\n';
        }
    }

    template <typename Function>
    void expectThrows(
        const std::string &test_name,
        Function &&function
    )
    {
        bool threw = false;

        try
        {
            function();
        }
        catch (...)
        {
            threw = true;
        }

        expect(threw, test_name);
    }

    bool vectorsNear(
        const std::vector<float> &actual,
        const std::vector<float> &expected,
        float tolerance
    )
    {
        if (actual.size() != expected.size())
        {
            return false;
        }

        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            if (std::abs(actual[i] - expected[i]) > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    int finish()
    {
        if (failureCount() == 0)
        {
            std::cout << "ALL TESTS PASSED\n";
            return 0;
        }

        std::cerr << failureCount() << " TEST(S) FAILED\n";
        return 1;
    }
}