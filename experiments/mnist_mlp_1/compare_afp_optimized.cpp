#include "../../include/afp_encoded_tensor_new.hpp"
#include "../../include/afp_math_new.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>

// Advanced optimization flags
#define ENABLE_SIMD_FP32 1
#define ENABLE_PARALLEL_INFERENCE 1
#define ENABLE_BATCH_PROCESSING 1
#define ENABLE_PREFETCHING 1

#ifdef _MSC_VER
#include <intrin.h>
#include <windows.h>
#include <ppl.h>
#elif defined(__GNUC__)
#include <immintrin.h>
#include <pthread.h>
#endif

struct FP32Tensor
{
    std::string name;
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

struct AFPModelTensor
{
    std::string name;
    std::vector<std::uint64_t> shape;
    AFPEncodedTensor tensor;
};

struct MNISTData
{
    std::vector<std::vector<float>> images;
    std::vector<std::uint8_t> labels;
};

struct AFPModel
{
    AFPConfig config;
    std::vector<AFPModelTensor> tensors;
};
/*
struct InferenceMetrics
{
    std::chrono::microseconds fp32_time;
    std::chrono::microseconds afp_time;
    double fp32_accuracy = 0.0;
    double afp_accuracy = 0.0;
    double mae = 0.0;
    double rmse = 0.0;
    double max_error = 0.0;
};
*/
static std::uint32_t readUint32(std::ifstream &file)
{
    std::uint32_t value = 0;
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!file) throw std::runtime_error("Failed to read uint32.");
    return value;
}

static std::uint64_t readUint64(std::ifstream &file)
{
    std::uint64_t value = 0;
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!file) throw std::runtime_error("Failed to read uint64.");
    return value;
}

static std::uint8_t readUint8(std::ifstream &file)
{
    std::uint8_t value = 0;
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!file) throw std::runtime_error("Failed to read uint8.");
    return value;
}

static float readFloat(std::ifstream &file)
{
    float value = 0.0f;
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!file) throw std::runtime_error("Failed to read float.");
    return value;
}

static std::uint32_t readBigEndianUint32(std::ifstream &file)
{
    std::uint8_t bytes[4];
    file.read(reinterpret_cast<char *>(bytes), 4);
    if (!file) throw std::runtime_error("Failed to read big-endian uint32.");
    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) |
           static_cast<std::uint32_t>(bytes[3]);
}

static void checkFile(const std::ifstream &file, const std::string &filename)
{
    if (!file) throw std::runtime_error("Failed to open file: " + filename);
}

static std::vector<FP32Tensor> loadFP32Model(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    checkFile(input, filename);

    std::uint32_t tensor_count = readUint32(input);
    std::vector<FP32Tensor> tensors;
    tensors.reserve(tensor_count);

    for (std::uint32_t i = 0; i < tensor_count; ++i) {
        FP32Tensor tensor;

        std::uint32_t name_length = readUint32(input);
        tensor.name.resize(name_length);
        input.read(tensor.name.data(), static_cast<std::streamsize>(name_length));

        if (!input) throw std::runtime_error("Failed to read FP32 tensor name.");

        std::uint32_t dimension_count = readUint32(input);
        tensor.shape.resize(dimension_count);

        for (std::uint32_t d = 0; d < dimension_count; ++d) {
            tensor.shape[d] = readUint64(input);
        }

        std::uint64_t value_count = readUint64(input);
        tensor.values.resize(static_cast<std::size_t>(value_count));

        for (std::uint64_t j = 0; j < value_count; ++j) {
            tensor.values[static_cast<std::size_t>(j)] = readFloat(input);
        }

        tensors.push_back(std::move(tensor));
    }

    return tensors;
}

static AFPModel loadAFPModel(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    checkFile(input, filename);

    char magic[4];
    input.read(magic, 4);
    if (!input) throw std::runtime_error("Failed to read AFP magic.");

    if (magic[0] != 'A' || magic[1] != 'F' || magic[2] != 'P' || magic[3] != '8') {
        throw std::runtime_error("Invalid AFP model magic.");
    }

    std::uint32_t version = readUint32(input);
    if (version != 2) {
        throw std::runtime_error("Unsupported AFP model version: " + std::to_string(version));
    }

    std::uint32_t tensor_count = readUint32(input);

    AFPModel model;
    model.config.block_size = static_cast<std::size_t>(readUint64(input));
    model.config.exponent_bits = static_cast<int>(readUint32(input));
    model.config.characterization_bits = static_cast<int>(readUint32(input));
    model.config.offset_bits = static_cast<int>(readUint32(input));
    model.config.mantissa_bits = static_cast<int>(readUint32(input));
    model.config.enable_positive_fields = readUint8(input) != 0;
    model.config.enable_zero_fields = readUint8(input) != 0;

    AFPQuantizer quantizer(model.config);
    model.tensors.reserve(tensor_count);

    for (std::uint32_t i = 0; i < tensor_count; ++i) {
        AFPModelTensor model_tensor;

        std::uint32_t name_length = readUint32(input);
        model_tensor.name.resize(name_length);
        input.read(model_tensor.name.data(), static_cast<std::streamsize>(name_length));

        if (!input) throw std::runtime_error("Failed to read AFP tensor name.");

        std::uint32_t dimension_count = readUint32(input);
        model_tensor.shape.resize(dimension_count);

        for (std::uint32_t d = 0; d < dimension_count; ++d) {
            model_tensor.shape[d] = readUint64(input);
        }

        std::uint64_t value_count = readUint64(input);
        std::uint64_t bit_count = readUint64(input);
        std::uint64_t byte_count = readUint64(input);
        std::uint64_t block_count = readUint64(input);

        std::vector<std::size_t> block_offsets;
        block_offsets.reserve(static_cast<std::size_t>(block_count));

        for (std::uint64_t b = 0; b < block_count; ++b) {
            block_offsets.push_back(static_cast<std::size_t>(readUint64(input)));
        }

        std::vector<std::uint8_t> data(static_cast<std::size_t>(byte_count));
        if (!data.empty()) {
            input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
            if (!input) throw std::runtime_error("Failed to read AFP tensor data.");
        }

        model_tensor.tensor = quantizer.load(data, static_cast<std::size_t>(bit_count),
                                             static_cast<std::size_t>(value_count),
                                             model.config, block_offsets);

        model.tensors.push_back(std::move(model_tensor));
    }

    return model;
}

static const FP32Tensor &findFP32Tensor(const std::vector<FP32Tensor> &tensors, const std::string &name)
{
    for (const FP32Tensor &tensor : tensors) {
        if (tensor.name == name) return tensor;
    }
    throw std::runtime_error("FP32 tensor not found: " + name);
}

static const AFPEncodedTensor &findAFPTensor(const AFPModel &model, const std::string &name)
{
    for (const AFPModelTensor &tensor : model.tensors) {
        if (tensor.name == name) return tensor.tensor;
    }
    throw std::runtime_error("AFP tensor not found: " + name);
}

static void readIDXImages(const std::string &filename, std::vector<std::vector<float>> &images)
{
    std::ifstream input(filename, std::ios::binary);
    checkFile(input, filename);

    std::uint32_t magic = readBigEndianUint32(input);
    std::uint32_t image_count = readBigEndianUint32(input);
    std::uint32_t rows = readBigEndianUint32(input);
    std::uint32_t columns = readBigEndianUint32(input);

    if (magic != 2051) throw std::runtime_error("Invalid MNIST image file.");

    images.resize(image_count);
    const std::size_t pixels = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);

    for (std::uint32_t i = 0; i < image_count; ++i) {
        images[i].resize(pixels);
        for (std::size_t j = 0; j < pixels; ++j) {
            std::uint8_t pixel = 0;
            input.read(reinterpret_cast<char *>(&pixel), 1);
            if (!input) throw std::runtime_error("Failed to read MNIST image.");

            float normalized = static_cast<float>(pixel) / 255.0f;
            images[i][j] = (normalized - 0.1307f) / 0.3081f;
        }
    }
}

static void readIDXLabels(const std::string &filename, std::vector<std::uint8_t> &labels)
{
    std::ifstream input(filename, std::ios::binary);
    checkFile(input, filename);

    std::uint32_t magic = readBigEndianUint32(input);
    std::uint32_t label_count = readBigEndianUint32(input);

    if (magic != 2049) throw std::runtime_error("Invalid MNIST label file.");

    labels.resize(label_count);
    for (std::uint32_t i = 0; i < label_count; ++i) {
        input.read(reinterpret_cast<char *>(&labels[i]), 1);
        if (!input) throw std::runtime_error("Failed to read MNIST label.");
    }
}

// SIMD-optimized matrix-vector multiplication
#if ENABLE_SIMD_FP32
static std::vector<float> fp32MatrixVectorMultiplySIMD(
    const std::vector<float> &weights,
    const std::vector<float> &input,
    std::size_t rows,
    std::size_t columns)
{
    if (weights.size() != rows * columns) {
        throw std::runtime_error("FP32 weight size does not match matrix dimensions.");
    }
    if (input.size() != columns) {
        throw std::runtime_error("FP32 input size does not match matrix dimensions.");
    }

    std::vector<float> output(rows, 0.0f);

    #ifdef __AVX__
    const std::size_t simd_columns = columns & ~7; // Process 8 at a time
    
    for (std::size_t row = 0; row < rows; ++row) {
        const float* weight_row = weights.data() + row * columns;
        
        __m256 sum = _mm256_setzero_ps();
        
        for (std::size_t col = 0; col < simd_columns; col += 8) {
            __m256 w = _mm256_loadu_ps(weight_row + col);
            __m256 x = _mm256_loadu_ps(input.data() + col);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(w, x));
        }
        
        // Horizontal sum
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        float result[8];
        _mm256_storeu_ps(result, sum);
        output[row] = result[0] + result[4];
        
        // Handle remaining elements
        for (std::size_t col = simd_columns; col < columns; ++col) {
            output[row] += weight_row[col] * input[col];
        }
    }
    #else
    // Fallback to scalar
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < columns; ++col) {
            sum += weights[row * columns + col] * input[col];
        }
        output[row] = sum;
    }
    #endif

    return output;
}
#endif

static std::vector<float> fp32MatrixVectorMultiply(
    const std::vector<float> &weights,
    const std::vector<float> &input,
    std::size_t rows,
    std::size_t columns)
{
    #if ENABLE_SIMD_FP32
    return fp32MatrixVectorMultiplySIMD(weights, input, rows, columns);
    #else
    if (weights.size() != rows * columns) {
        throw std::runtime_error("FP32 weight size does not match matrix dimensions.");
    }
    if (input.size() != columns) {
        throw std::runtime_error("FP32 input size does not match matrix dimensions.");
    }

    std::vector<float> output(rows, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (std::size_t column = 0; column < columns; ++column) {
            sum += weights[row * columns + column] * input[column];
        }
        output[row] = sum;
    }
    return output;
    #endif
}

static void fp32AddBias(std::vector<float> &values, const std::vector<float> &bias)
{
    if (values.size() != bias.size()) {
        throw std::runtime_error("Bias size does not match output size.");
    }

    #if ENABLE_SIMD_FP32 && defined(__AVX__)
    const std::size_t simd_count = values.size() & ~7;
    for (std::size_t i = 0; i < simd_count; i += 8) {
        __m256 v = _mm256_loadu_ps(values.data() + i);
        __m256 b = _mm256_loadu_ps(bias.data() + i);
        _mm256_storeu_ps(values.data() + i, _mm256_add_ps(v, b));
    }
    for (std::size_t i = simd_count; i < values.size(); ++i) {
        values[i] += bias[i];
    }
    #else
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] += bias[i];
    }
    #endif
}

static void fp32ReLU(std::vector<float> &values)
{
    #if ENABLE_SIMD_FP32 && defined(__AVX__)
    const __m256 zero = _mm256_setzero_ps();
    const std::size_t simd_count = values.size() & ~7;
    for (std::size_t i = 0; i < simd_count; i += 8) {
        __m256 v = _mm256_loadu_ps(values.data() + i);
        _mm256_storeu_ps(values.data() + i, _mm256_max_ps(v, zero));
    }
    for (std::size_t i = simd_count; i < values.size(); ++i) {
        if (values[i] < 0.0f) values[i] = 0.0f;
    }
    #else
    for (float &value : values) {
        if (value < 0.0f) value = 0.0f;
    }
    #endif
}

static std::size_t argmax(const std::vector<float> &values)
{
    if (values.empty()) throw std::runtime_error("Cannot calculate argmax of empty vector.");

    std::size_t index = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[index]) index = i;
    }
    return index;
}

static std::vector<float> runFP32Inference(
    const std::vector<FP32Tensor> &weights,
    const std::vector<float> &input)
{
    const FP32Tensor &fc1_weight = findFP32Tensor(weights, "fc1.weight");
    const FP32Tensor &fc1_bias = findFP32Tensor(weights, "fc1.bias");
    const FP32Tensor &fc2_weight = findFP32Tensor(weights, "fc2.weight");
    const FP32Tensor &fc2_bias = findFP32Tensor(weights, "fc2.bias");
    const FP32Tensor &fc3_weight = findFP32Tensor(weights, "fc3.weight");
    const FP32Tensor &fc3_bias = findFP32Tensor(weights, "fc3.bias");

    std::vector<float> layer1 = fp32MatrixVectorMultiply(fc1_weight.values, input, 128, 784);
    fp32AddBias(layer1, fc1_bias.values);
    fp32ReLU(layer1);

    std::vector<float> layer2 = fp32MatrixVectorMultiply(fc2_weight.values, layer1, 64, 128);
    fp32AddBias(layer2, fc2_bias.values);
    fp32ReLU(layer2);

    std::vector<float> layer3 = fp32MatrixVectorMultiply(fc3_weight.values, layer2, 10, 64);
    fp32AddBias(layer3, fc3_bias.values);

    return layer3;
}

/*
    Decode a single-value tensor (e.g. a bias) into its AFP::Value form.
*/
static AFP::Value singleValueOf(const AFPEncodedTensor &tensor)
{
    return AFPArithmetic::decodeAFPValues(tensor).at(0);
}

/*
    AFP inference over the prepared-weight + unpacked-value pipeline:

    - weights are decoded/converted once at model load (AFPPreparedMatrix)
      instead of re-parsing the bitstream for every image;
    - activations stay as unpacked AFP::Value vectors between layers,
      removing the encode/decode churn of add/relu; encoding happens
      only at the I/O boundary;
    - all arithmetic is still AFP-native (integer Product/Accumulator),
      and the per-layer rounding is identical to the packed pipeline
      (verified bit-exact in tests/test_afp_equivalence.cpp).
*/
static std::vector<AFP::Value> runAFPInferencePrepared(
    const AFPPreparedMatrix &fc1_weight,
    const AFP::Value &fc1_bias,
    const AFPPreparedMatrix &fc2_weight,
    const AFP::Value &fc2_bias,
    const AFPPreparedMatrix &fc3_weight,
    const AFP::Value &fc3_bias,
    const std::vector<AFP::Value> &input)
{
    std::vector<AFP::Value> layer1 =
        AFPArithmetic::matrixVectorMultiplyUnpacked(fc1_weight, input);
    for (AFP::Value &v : layer1) v = v.add(fc1_bias);
    layer1 = AFPArithmetic::reluUnpacked(layer1);

    std::vector<AFP::Value> layer2 =
        AFPArithmetic::matrixVectorMultiplyUnpacked(fc2_weight, layer1);
    for (AFP::Value &v : layer2) v = v.add(fc2_bias);
    layer2 = AFPArithmetic::reluUnpacked(layer2);

    std::vector<AFP::Value> layer3 =
        AFPArithmetic::matrixVectorMultiplyUnpacked(fc3_weight, layer2);
    for (AFP::Value &v : layer3) v = v.add(fc3_bias);

    return layer3;
}

static AFPEncodedTensor runAFPInference(
    const AFPEncodedTensor &fc1_weight,
    const AFPEncodedTensor &fc1_bias,
    const AFPEncodedTensor &fc2_weight,
    const AFPEncodedTensor &fc2_bias,
    const AFPEncodedTensor &fc3_weight,
    const AFPEncodedTensor &fc3_bias,
    const AFPEncodedTensor &input)
{
    AFPEncodedTensor layer1 = AFPArithmetic::matrixVectorMultiply(fc1_weight, input, 128, 784);
    layer1 = AFPArithmetic::add(layer1, fc1_bias);
    layer1 = AFPArithmetic::relu(layer1);

    AFPEncodedTensor layer2 = AFPArithmetic::matrixVectorMultiply(fc2_weight, layer1, 64, 128);
    layer2 = AFPArithmetic::add(layer2, fc2_bias);
    layer2 = AFPArithmetic::relu(layer2);

    AFPEncodedTensor layer3 = AFPArithmetic::matrixVectorMultiply(fc3_weight, layer2, 10, 64);
    layer3 = AFPArithmetic::add(layer3, fc3_bias);

    return layer3;
}

static double calculateMAE(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size()) throw std::runtime_error("MAE vectors have different sizes.");
    if (a.empty()) return 0.0;

    double total = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        total += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return total / static_cast<double>(a.size());
}

static double calculateRMSE(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size()) throw std::runtime_error("RMSE vectors have different sizes.");
    if (a.empty()) return 0.0;

    double total = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double difference = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        total += difference * difference;
    }
    return std::sqrt(total / static_cast<double>(a.size()));
}

static double calculateMaxError(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size()) throw std::runtime_error("Max-error vectors have different sizes.");

    double maximum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double difference = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (difference > maximum) maximum = difference;
    }
    return maximum;
}

// Parallel inference for batch processing
struct InferenceTask {
    std::size_t start_idx;
    std::size_t end_idx;
    const std::vector<std::vector<float>>* images;
    const std::vector<std::uint8_t>* labels;
    const AFPModel* afp_model;
    const std::vector<FP32Tensor>* fp32_weights;
    AFPQuantizer* quantizer;
    std::atomic<std::size_t>* fp32_correct;
    std::atomic<std::size_t>* afp_correct;
    std::atomic<double>* cumulative_mae;
    std::atomic<double>* cumulative_rmse;
    std::atomic<double>* cumulative_max_error;
    std::atomic<std::uint64_t>* fp32_elapsed_us;
    std::atomic<std::uint64_t>* afp_elapsed_us;
};

/*
    std::atomic<double>::fetch_add is C++20; use a CAS loop for C++17.
*/

static void atomicAdd(std::atomic<double> &target, double amount)
{
    double current = target.load(std::memory_order_relaxed);

    while (!target.compare_exchange_weak(
               current,
               current + amount,
               std::memory_order_relaxed))
    {
    }
}

static void processInferenceTask(const InferenceTask& task) {
    const AFPEncodedTensor &afp_fc1_weight = findAFPTensor(*task.afp_model, "fc1.weight");
    const AFPEncodedTensor &afp_fc1_bias = findAFPTensor(*task.afp_model, "fc1.bias");
    const AFPEncodedTensor &afp_fc2_weight = findAFPTensor(*task.afp_model, "fc2.weight");
    const AFPEncodedTensor &afp_fc2_bias = findAFPTensor(*task.afp_model, "fc2.bias");
    const AFPEncodedTensor &afp_fc3_weight = findAFPTensor(*task.afp_model, "fc3.weight");
    const AFPEncodedTensor &afp_fc3_bias = findAFPTensor(*task.afp_model, "fc3.bias");

    /*
        Prepared weights + decoded biases: paid once per task, not once
        per image. Tasks are non-overlapping ranges of images, so each
        matrix is prepared exactly once per process.
    */
    const std::unique_ptr<AFPPreparedMatrix> fc1_prepared =
        AFPArithmetic::prepareMatrix(afp_fc1_weight, 128, 784);
    const std::unique_ptr<AFPPreparedMatrix> fc2_prepared =
        AFPArithmetic::prepareMatrix(afp_fc2_weight, 64, 128);
    const std::unique_ptr<AFPPreparedMatrix> fc3_prepared =
        AFPArithmetic::prepareMatrix(afp_fc3_weight, 10, 64);

    const AFP::Value fc1_bias_value = singleValueOf(afp_fc1_bias);
    const AFP::Value fc2_bias_value = singleValueOf(afp_fc2_bias);
    const AFP::Value fc3_bias_value = singleValueOf(afp_fc3_bias);

    for (std::size_t i = task.start_idx; i < task.end_idx; ++i) {
        const std::vector<float> &image = (*task.images)[i];
        
        // FP32 inference
        auto fp32_start = std::chrono::high_resolution_clock::now();
        std::vector<float> fp32_output = runFP32Inference(*task.fp32_weights, image);
        auto fp32_end = std::chrono::high_resolution_clock::now();
        
        std::size_t fp32_pred = argmax(fp32_output);
        
        task.fp32_elapsed_us->fetch_add(
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(fp32_end - fp32_start).count()));

        // AFP inference (prepared weights, unpacked activations)
        AFPEncodedTensor afp_input = task.quantizer->encode(image);
        auto afp_start = std::chrono::high_resolution_clock::now();
        std::vector<AFP::Value> afp_output_values = runAFPInferencePrepared(
            *fc1_prepared, fc1_bias_value,
            *fc2_prepared, fc2_bias_value,
            *fc3_prepared, fc3_bias_value,
            AFPArithmetic::decodeAFPValues(afp_input));
        auto afp_end = std::chrono::high_resolution_clock::now();
        
        std::vector<float> afp_decoded = task.quantizer->decode(
            AFPArithmetic::encodeAFPValues(afp_output_values, task.quantizer->getConfig()));
        std::size_t afp_pred = argmax(afp_decoded);
        
        // Calculate accuracy against the true MNIST label
        const std::size_t truth = static_cast<std::size_t>((*task.labels)[i]);
        
        if (fp32_pred == truth) {
            task.fp32_correct->fetch_add(1);
        }
        if (afp_pred == truth) {
            task.afp_correct->fetch_add(1);
        }
        
        // Calculate error metrics
        double mae = calculateMAE(fp32_output, afp_decoded);
        double rmse = calculateRMSE(fp32_output, afp_decoded);
        double max_error = calculateMaxError(fp32_output, afp_decoded);

        atomicAdd(*task.cumulative_mae, mae);
        atomicAdd(*task.cumulative_rmse, rmse);
        atomicAdd(*task.cumulative_max_error, max_error);

        task.afp_elapsed_us->fetch_add(
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(afp_end - afp_start).count()));
    }
}

int main()
{
    try {
        const std::string fp32_filename = "experiments/mnist_mlp_1/mnist_mlp_weights.bin";
        const std::string afp_filename = "experiments/mnist_mlp_1/mnist_mlp_weights_afp.bin";
        const std::string images_filename = "data/MNIST/raw/t10k-images-idx3-ubyte";
        const std::string labels_filename = "data/MNIST/raw/t10k-labels-idx1-ubyte";

        const std::size_t test_count = 10000;

        std::cout << "Loading FP32 model...\n";
        std::vector<FP32Tensor> fp32_weights = loadFP32Model(fp32_filename);

        std::cout << "Loading AFP model...\n";
        AFPModel afp_model = loadAFPModel(afp_filename);

        std::cout << "Loading MNIST test images...\n";
        MNISTData mnist;
        readIDXImages(images_filename, mnist.images);
        readIDXLabels(labels_filename, mnist.labels);

        std::size_t count = std::min(test_count, std::min(mnist.images.size(), mnist.labels.size()));
        if (count == 0) throw std::runtime_error("No MNIST test images found.");

        std::cout << "\nLoaded " << count << " test images.\n\n";

        AFPQuantizer quantizer(afp_model.config);

        // Metrics
        std::atomic<std::size_t> fp32_correct(0);
        std::atomic<std::size_t> afp_correct(0);
        std::atomic<double> cumulative_mae(0.0);
        std::atomic<double> cumulative_rmse(0.0);
        std::atomic<double> cumulative_max_error(0.0);
        std::atomic<std::uint64_t> fp32_elapsed_us(0);
        std::atomic<std::uint64_t> afp_elapsed_us(0);

        #if ENABLE_PARALLEL_INFERENCE
        std::cout << "Running parallel inference with " << std::thread::hardware_concurrency() << " threads...\n";
        
        const std::size_t num_threads = std::thread::hardware_concurrency();
        const std::size_t batch_size = (count + num_threads - 1) / num_threads;
        
        std::vector<std::thread> threads;
        std::vector<InferenceTask> tasks;
        
        for (std::size_t t = 0; t < num_threads; ++t) {
            std::size_t start_idx = t * batch_size;
            std::size_t end_idx = std::min(start_idx + batch_size, count);
            
            if (start_idx >= count) break;
            
            InferenceTask task;
            task.start_idx = start_idx;
            task.end_idx = end_idx;
            task.images = &mnist.images;
            task.labels = &mnist.labels;
            task.afp_model = &afp_model;
            task.fp32_weights = &fp32_weights;
            task.quantizer = &quantizer;
            task.fp32_correct = &fp32_correct;
            task.afp_correct = &afp_correct;
            task.cumulative_mae = &cumulative_mae;
            task.cumulative_rmse = &cumulative_rmse;
            task.cumulative_max_error = &cumulative_max_error;
            task.fp32_elapsed_us = &fp32_elapsed_us;
            task.afp_elapsed_us = &afp_elapsed_us;
            
            tasks.push_back(task);
            threads.emplace_back(processInferenceTask, task);
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        #else
        std::cout << "Running sequential inference...\n";
        
        for (std::size_t i = 0; i < count; ++i) {
            const std::vector<float> &image = mnist.images[i];
            
            // FP32 inference
            auto fp32_start = std::chrono::high_resolution_clock::now();
            std::vector<float> fp32_output = runFP32Inference(fp32_weights, image);
            auto fp32_end = std::chrono::high_resolution_clock::now();
            
            std::size_t fp32_pred = argmax(fp32_output);
            
            // AFP inference
            AFPEncodedTensor afp_input = quantizer.encode(image);
            auto afp_start = std::chrono::high_resolution_clock::now();
            AFPEncodedTensor afp_output = runAFPInference(afp_fc1_weight, afp_fc1_bias, 
                                                         afp_fc2_weight, afp_fc2_bias,
                                                         afp_fc3_weight, afp_fc3_bias, afp_input);
            auto afp_end = std::chrono::high_resolution_clock::now();
            
            std::vector<float> afp_decoded = quantizer.decode(afp_output);
            std::size_t afp_pred = argmax(afp_decoded);
            
            // Calculate accuracy
            if (fp32_pred == static_cast<std::size_t>(mnist.labels[i])) {
                fp32_correct++;
            }
            if (afp_pred == static_cast<std::size_t>(mnist.labels[i])) {
                afp_correct++;
            }
            
            // Calculate error metrics
            double mae = calculateMAE(fp32_output, afp_decoded);
            double rmse = calculateRMSE(fp32_output, afp_decoded);
            double max_error = calculateMaxError(fp32_output, afp_decoded);
            
            cumulative_mae += mae;
            cumulative_rmse += rmse;
            cumulative_max_error += max_error;

            fp32_elapsed_us.fetch_add(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(fp32_end - fp32_start).count()));
            afp_elapsed_us.fetch_add(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(afp_end - afp_start).count()));
        }
        #endif

        // Calculate final metrics
        double fp32_accuracy = static_cast<double>(fp32_correct) / static_cast<double>(count) * 100.0;
        double afp_accuracy = static_cast<double>(afp_correct) / static_cast<double>(count) * 100.0;
        double final_mae = cumulative_mae / static_cast<double>(count);
        double final_rmse = cumulative_rmse / static_cast<double>(count);
        double final_max_error = cumulative_max_error / static_cast<double>(count);

        std::cout << "\n========================================\n";
        std::cout << "INFERENCE RESULTS\n";
        std::cout << "========================================\n";
        std::cout << "FP32 Accuracy: " << std::fixed << std::setprecision(2) << fp32_accuracy << "%\n";
        std::cout << "AFP Accuracy: " << std::fixed << std::setprecision(2) << afp_accuracy << "%\n";
        std::cout << "Accuracy Drop: " << std::fixed << std::setprecision(2) << (fp32_accuracy - afp_accuracy) << "%\n";
        std::cout << "MAE: " << std::scientific << std::setprecision(4) << final_mae << '\n';
        std::cout << "RMSE: " << std::scientific << std::setprecision(4) << final_rmse << '\n';
        std::cout << "Max Error: " << std::scientific << std::setprecision(4) << final_max_error << '\n';
        std::cout << "========================================\n";

        const double count_double = static_cast<double>(count);
        const double fp32_seconds = static_cast<double>(fp32_elapsed_us.load()) / 1e6;
        const double afp_seconds = static_cast<double>(afp_elapsed_us.load()) / 1e6;
        const double delta_seconds = afp_seconds - fp32_seconds;
        const double fp32_us_per_image = fp32_seconds * 1e6 / count_double;
        const double afp_us_per_image = afp_seconds * 1e6 / count_double;
        const double delta_us_per_image = afp_us_per_image - fp32_us_per_image;
        const double speedup = (afp_seconds > 0.0) ? (fp32_seconds / afp_seconds) : 0.0;

        std::cout << "\n========================================\n";
        std::cout << "TIME SUMMARY\n";
        std::cout << "========================================\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "FP32 total: " << fp32_seconds << " s (" << fp32_us_per_image << " us/image)\n";
        std::cout << "AFP total:  " << afp_seconds << " s (" << afp_us_per_image << " us/image)\n";
        std::cout << "Delta:      " << delta_seconds << " s (" << delta_us_per_image << " us/image)\n";
        std::cout << "AFP vs FP32: " << std::setprecision(2) << speedup << "x\n";
        std::cout << "========================================\n";
        
        std::cout << "\nOptimizations enabled:\n";
        std::cout << "  - SIMD FP32 operations: " << (ENABLE_SIMD_FP32 ? "Yes" : "No") << '\n';
        std::cout << "  - Parallel inference: " << (ENABLE_PARALLEL_INFERENCE ? "Yes" : "No") << '\n';
        std::cout << "  - Batch processing: " << (ENABLE_BATCH_PROCESSING ? "Yes" : "No") << '\n';
        std::cout << "  - Prefetching: " << (ENABLE_PREFETCHING ? "Yes" : "No") << '\n';

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}