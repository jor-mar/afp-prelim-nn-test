#include "../../include/afp.hpp"
#include "../../include/encoded_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Tensor
{
    std::string name;
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

std::uint32_t readUInt32(std::ifstream &file)
{
    std::uint32_t value;

    file.read(
        reinterpret_cast<char *>(&value),
        sizeof(value));

    if (!file)
        throw std::runtime_error("Failed to read uint32");

    return value;
}

std::uint64_t readUInt64(std::ifstream &file)
{
    std::uint64_t value;

    file.read(
        reinterpret_cast<char *>(&value),
        sizeof(value));

    if (!file)
        throw std::runtime_error("Failed to read uint64");

    return value;
}

float readFloat(std::ifstream &file)
{
    float value;

    file.read(
        reinterpret_cast<char *>(&value),
        sizeof(value));

    if (!file)
        throw std::runtime_error("Failed to read float");

    return value;
}

std::string readString(std::ifstream &file)
{
    const std::uint32_t length = readUInt32(file);

    std::string value(length, '\0');

    if (length > 0)
    {
        file.read(
            &value[0],
            static_cast<std::streamsize>(length));
    }

    if (!file)
        throw std::runtime_error("Failed to read string");

    return value;
}

Tensor readTensor(std::ifstream &file)
{
    Tensor tensor;

    tensor.name = readString(file);

    const std::uint32_t dimension_count = readUInt32(file);

    tensor.shape.resize(dimension_count);

    for (std::uint32_t i = 0; i < dimension_count; ++i)
    {
        tensor.shape[i] = readUInt64(file);
    }

    const std::uint64_t value_count = readUInt64(file);

    tensor.values.resize(value_count);

    for (std::uint64_t i = 0; i < value_count; ++i)
    {
        tensor.values[i] = readFloat(file);
    }

    return tensor;
}

struct ErrorMetrics
{
    double mse = 0.0;
    double mae = 0.0;
    double rmse = 0.0;

    double max_error = 0.0;

    double mean_relative_error = 0.0;
    double max_relative_error = 0.0;

    std::size_t exact_count = 0;
    std::size_t relative_error_count = 0;

    std::size_t below_01_percent = 0;
    std::size_t below_1_percent = 0;
    std::size_t below_5_percent = 0;
    std::size_t below_10_percent = 0;

    std::size_t reconstructed_zero_count = 0;

    float original_min = 0.0f;
    float original_max = 0.0f;

    float reconstructed_min = 0.0f;
    float reconstructed_max = 0.0f;
};

ErrorMetrics calculateError(
    const std::vector<float> &original,
    const std::vector<float> &reconstructed)
{
    if (original.size() != reconstructed.size())
        throw std::runtime_error("Tensor size mismatch");

    ErrorMetrics metrics;

    if (original.empty())
        return metrics;

    metrics.original_min =
        *std::min_element(original.begin(), original.end());

    metrics.original_max =
        *std::max_element(original.begin(), original.end());

    metrics.reconstructed_min =
        *std::min_element(
            reconstructed.begin(),
            reconstructed.end());

    metrics.reconstructed_max =
        *std::max_element(
            reconstructed.begin(),
            reconstructed.end());

    constexpr double relative_error_threshold = 1e-6;

    for (std::size_t i = 0; i < original.size(); ++i)
    {
        const double original_value =
            static_cast<double>(original[i]);

        const double reconstructed_value =
            static_cast<double>(reconstructed[i]);

        const double error =
            std::abs(reconstructed_value - original_value);

        metrics.mse += error * error;
        metrics.mae += error;

        metrics.max_error =
            std::max(metrics.max_error, error);

        if (reconstructed_value == 0.0)
        {
            ++metrics.reconstructed_zero_count;
        }

        if (reconstructed[i] == original[i])
        {
            ++metrics.exact_count;
        }

        if (std::abs(original_value) >= relative_error_threshold)
        {
            const double relative_error =
                error / std::abs(original_value);

            metrics.mean_relative_error += relative_error;

            metrics.max_relative_error =
                std::max(
                    metrics.max_relative_error,
                    relative_error);

            ++metrics.relative_error_count;

            if (relative_error < 0.001)
                ++metrics.below_01_percent;

            if (relative_error < 0.01)
                ++metrics.below_1_percent;

            if (relative_error < 0.05)
                ++metrics.below_5_percent;

            if (relative_error < 0.10)
                ++metrics.below_10_percent;
        }
    }

    const double value_count =
        static_cast<double>(original.size());

    metrics.mse /= value_count;
    metrics.mae /= value_count;

    metrics.rmse = std::sqrt(metrics.mse);

    if (metrics.relative_error_count > 0)
    {
        metrics.mean_relative_error /=
            static_cast<double>(metrics.relative_error_count);
    }

    return metrics;
}

void printShape(const std::vector<std::uint64_t> &shape)
{
    std::cout << '(';

    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i];

        if (i + 1 < shape.size())
            std::cout << ", ";
    }

    std::cout << ')';
}

double percentage(
    std::size_t count,
    std::size_t total)
{
    if (total == 0)
        return 0.0;

    return 100.0 *
           static_cast<double>(count) /
           static_cast<double>(total);
}

int main()
{
    try
    {
        std::ifstream file(
            "mnist_mlp_weights.bin",
            std::ios::binary);

        if (!file)
        {
            throw std::runtime_error(
                "Could not open mnist_mlp_weights.bin");
        }

        const std::uint32_t tensor_count =
            readUInt32(file);

        std::vector<Tensor> tensors;
        tensors.reserve(tensor_count);

        for (std::uint32_t i = 0; i < tensor_count; ++i)
        {
            tensors.push_back(readTensor(file));
        }

        AFPConfig config;
        AFPQuantizer quantizer(config);

        std::cout << std::fixed << std::setprecision(8);

        std::cout << "AFP8 Weight Benchmark\n";
        std::cout << "=====================\n\n";

        double total_original_bits = 0.0;
        double total_afp_bits = 0.0;

        double total_squared_error = 0.0;
        double total_absolute_error = 0.0;

        std::size_t total_values = 0;
        std::size_t total_exact = 0;
        std::size_t total_relative_error_count = 0;

        std::size_t total_below_01_percent = 0;
        std::size_t total_below_1_percent = 0;
        std::size_t total_below_5_percent = 0;
        std::size_t total_below_10_percent = 0;

        std::size_t total_reconstructed_zero = 0;

        for (const Tensor &tensor : tensors)
        {
            const AFPEncodedTensor encoded =
                quantizer.encode(tensor.values);

            const std::vector<float> reconstructed =
                quantizer.decode(encoded);

            const ErrorMetrics metrics =
                calculateError(
                    tensor.values,
                    reconstructed);

            const std::size_t value_count =
                tensor.values.size();

            const double original_bits =
                static_cast<double>(value_count) * 32.0;

            const double afp_bits =
                static_cast<double>(encoded.bitSize());

            const double bits_per_value =
                afp_bits /
                static_cast<double>(value_count);

            const double compression_ratio =
                original_bits / afp_bits;

            std::cout << tensor.name << '\n';

            std::cout << "  Shape: ";
            printShape(tensor.shape);
            std::cout << '\n';

            std::cout << "  Values: "
                      << value_count
                      << '\n';

            std::cout << "  Original: "
                      << original_bits
                      << " bits\n";

            std::cout << "  AFP8: "
                      << afp_bits
                      << " bits\n";

            std::cout << "  Bits/value: "
                      << bits_per_value
                      << '\n';

            std::cout << "  Compression: "
                      << compression_ratio
                      << "x\n";

            std::cout << '\n';

            std::cout << "  Original range: ["
                      << metrics.original_min
                      << ", "
                      << metrics.original_max
                      << "]\n";

            std::cout << "  Reconstructed range: ["
                      << metrics.reconstructed_min
                      << ", "
                      << metrics.reconstructed_max
                      << "]\n";

            std::cout << '\n';

            std::cout << "  MSE: "
                      << metrics.mse
                      << '\n';

            std::cout << "  RMSE: "
                      << metrics.rmse
                      << '\n';

            std::cout << "  MAE: "
                      << metrics.mae
                      << '\n';

            std::cout << "  Max error: "
                      << metrics.max_error
                      << '\n';

            std::cout << '\n';

            std::cout << "  Mean relative error: "
                      << metrics.mean_relative_error * 100.0
                      << "%\n";

            std::cout << "  Max relative error: "
                      << metrics.max_relative_error * 100.0
                      << "%\n";

            std::cout << '\n';

            std::cout << "  Exact reconstruction: "
                      << percentage(
                             metrics.exact_count,
                             value_count)
                      << "%\n";

            std::cout << "  Relative error < 0.1%: "
                      << percentage(
                             metrics.below_01_percent,
                             metrics.relative_error_count)
                      << "%\n";

            std::cout << "  Relative error < 1%: "
                      << percentage(
                             metrics.below_1_percent,
                             metrics.relative_error_count)
                      << "%\n";

            std::cout << "  Relative error < 5%: "
                      << percentage(
                             metrics.below_5_percent,
                             metrics.relative_error_count)
                      << "%\n";

            std::cout << "  Relative error < 10%: "
                      << percentage(
                             metrics.below_10_percent,
                             metrics.relative_error_count)
                      << "%\n";

            std::cout << '\n';

            std::cout << "  Reconstructed as zero: "
                      << percentage(
                             metrics.reconstructed_zero_count,
                             value_count)
                      << "%\n";

            std::cout << '\n';

            total_original_bits += original_bits;
            total_afp_bits += afp_bits;

            total_squared_error +=
                metrics.mse *
                static_cast<double>(value_count);

            total_absolute_error +=
                metrics.mae *
                static_cast<double>(value_count);

            total_values += value_count;

            total_exact += metrics.exact_count;

            total_relative_error_count +=
                metrics.relative_error_count;

            total_below_01_percent +=
                metrics.below_01_percent;

            total_below_1_percent +=
                metrics.below_1_percent;

            total_below_5_percent +=
                metrics.below_5_percent;

            total_below_10_percent +=
                metrics.below_10_percent;

            total_reconstructed_zero +=
                metrics.reconstructed_zero_count;
        }

        const double total_mse =
            total_squared_error /
            static_cast<double>(total_values);

        const double total_rmse =
            std::sqrt(total_mse);

        const double total_mae =
            total_absolute_error /
            static_cast<double>(total_values);

        std::cout << "=====================\n";
        std::cout << "TOTAL\n";
        std::cout << "=====================\n";

        std::cout << "Values: "
                  << total_values
                  << '\n';

        std::cout << "Original: "
                  << total_original_bits
                  << " bits\n";

        std::cout << "AFP8: "
                  << total_afp_bits
                  << " bits\n";

        std::cout << "Bits/value: "
                  << total_afp_bits /
                         static_cast<double>(total_values)
                  << '\n';

        std::cout << "Compression: "
                  << total_original_bits /
                         total_afp_bits
                  << "x\n";

        std::cout << '\n';

        std::cout << "MSE: "
                  << total_mse
                  << '\n';

        std::cout << "RMSE: "
                  << total_rmse
                  << '\n';

        std::cout << "MAE: "
                  << total_mae
                  << '\n';

        std::cout << '\n';

        std::cout << "Exact reconstruction: "
                  << percentage(
                         total_exact,
                         total_values)
                  << "%\n";

        std::cout << "Relative error < 0.1%: "
                  << percentage(
                         total_below_01_percent,
                         total_relative_error_count)
                  << "%\n";

        std::cout << "Relative error < 1%: "
                  << percentage(
                         total_below_1_percent,
                         total_relative_error_count)
                  << "%\n";

        std::cout << "Relative error < 5%: "
                  << percentage(
                         total_below_5_percent,
                         total_relative_error_count)
                  << "%\n";

        std::cout << "Relative error < 10%: "
                  << percentage(
                         total_below_10_percent,
                         total_relative_error_count)
                  << "%\n";

        std::cout << '\n';

        std::cout << "Reconstructed as zero: "
                  << percentage(
                         total_reconstructed_zero,
                         total_values)
                  << "%\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "Error: "
                  << error.what()
                  << '\n';

        return 1;
    }

    return 0;
}