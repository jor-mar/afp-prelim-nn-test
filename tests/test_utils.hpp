#pragma once

#include <cmath>
#include <cstddef>
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

    std::vector<std::string> &failedTests()
    {
        static std::vector<std::string> failures;
        return failures;
    }

    void expect(
        bool condition,
        const std::string &test_name)
    {
        if (!condition)
        {
            ++failureCount();
            failedTests().push_back(test_name);
        }
    }

    void expectNear(
        double actual,
        double expected,
        double tolerance,
        const std::string &test_name)
    {
        const bool passed = std::abs(actual - expected) <= tolerance;

        if (!passed)
        {
            ++failureCount();
            failedTests().push_back(test_name);

            std::cerr << "[FAIL] " << test_name << '\n';
            std::cerr << "    Expected: " << expected << '\n';
            std::cerr << "    Actual:   " << actual << '\n';
            std::cerr << "    Tolerance: " << tolerance << '\n';
        }
        else {
            std::cout << "[PASS]" << test_name << '\n';
        }
    }

    template <typename Function>
    void expectThrows(
        const std::string &test_name,
        Function &&function)
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
        float tolerance)
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
        std::cout << "\n========================================\n";

        if (failureCount() == 0)
        {
            std::cout << "ALL TESTS PASSED\n";
            std::cout << "========================================\n";
            return 0;
        }

        std::cout << failureCount() << " TEST(S) FAILED\n\n";

        std::cout << "Failed tests:\n";

        for (const std::string &test_name : failedTests())
        {
            std::cout << "  - " << test_name << '\n';
        }

        std::cout << "========================================\n";

        return 1;
    }
}