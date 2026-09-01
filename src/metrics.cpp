#include "../include/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

Metrics calculateMetrics(
    const std::vector<float> &original,
    const std::vector<float> &reconstructed,
    double epsilon)
{
    if (original.size() != reconstructed.size())
    {
        throw std::invalid_argument("Original and reconstructed tensors must have the same size");
    }

    if (epsilon <= 0.0)
    {
        throw std::invalid_argument("epsilon must be greater than zero");
    }

    Metrics metrics;

    metrics.total_values = original.size();

    if (original.empty())
    {
        return metrics;
    }

    double squared_error_sum = 0.0;
    double absolute_error_sum = 0.0;
    double relative_error_sum = 0.0;

    for (std::size_t i = 0; i < original.size();++i)
    {
        const double original_value = static_cast<double>(original[i]);

        const double reconstructed_value = static_cast<double>(reconstructed[i]);

        const double error = reconstructed_value - original_value;

        const double absolute_error = std::abs(error);

        squared_error_sum += error * error;

        absolute_error_sum += absolute_error;

        metrics.max_error = std::max(metrics.max_error, absolute_error);

        relative_error_sum += absolute_error / (std::abs(original_value) + epsilon);
    }

    const double count = static_cast<double>(original.size());

    metrics.mse = squared_error_sum / count;

    metrics.mae = absolute_error_sum / count;

    metrics.mean_relative_error = relative_error_sum / count;

    return metrics;
}