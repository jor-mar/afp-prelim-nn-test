#pragma once

#include <cstddef>
#include <vector>

struct Metrics
{
    double mse = 0.0;
    double mae = 0.0;
    double max_error = 0.0;
    double mean_relative_error = 0.0;
    std::size_t total_values = 0;
};

Metrics calculateMetrics(
    const std::vector<float>& original,
    const std::vector<float>& reconstructed,
    double epsilon = 1e-12
);
