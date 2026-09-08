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

// ---------------------------------------------------------------------------
// File resolution
//
// Inputs are looked up next to the executable first, then in the current
// directory, so the tools behave identically whether they are started from
// the repository root, the experiment directory, or anywhere else.
// ---------------------------------------------------------------------------

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__unix__) || defined(__linux__)
#include <unistd.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#endif

static std::string executablePath()
{
#if defined(_WIN32)
    char buffer[1024];
    const DWORD length = GetModuleFileNameA(nullptr, buffer, static_cast<DWORD>(sizeof(buffer)));
    if (length == 0 || length >= sizeof(buffer)) return std::string();
    return std::string(buffer, length);
#elif defined(__linux__) || defined(__unix__)
    char buffer[PATH_MAX + 1];
    const ssize_t length = readlink("/proc/self/exe", buffer, PATH_MAX);
    if (length <= 0) return std::string();
    return std::string(buffer, static_cast<std::size_t>(length));
#elif defined(__APPLE__)
    uint32_t size = 0;
    if (_NSGetExecutablePath(nullptr, &size) != 0) return std::string();
    std::vector<char> buffer(size + 1);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) return std::string();
    return std::string(buffer.data());
#else
    return std::string();
#endif
}

static std::string directoryOf(const std::string &path)
{
    const std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return std::string();
    return path.substr(0, slash);
}

static std::string stemOf(const std::string &path)
{
    const std::size_t start = path.find_last_of("/\\");
    const std::size_t name_start = (start == std::string::npos) ? 0 : start + 1;
    const std::size_t dot = path.find_last_of('.');
    const std::size_t name_end = (dot == std::string::npos || dot < name_start) ? path.size() : dot;
    return path.substr(name_start, name_end - name_start);
}

static std::string joinPath(const std::string &directory, const std::string &file)
{
    if (directory.empty()) return file;
    const char last = directory.back();
    if (last == '/' || last == '\\') return directory + file;
    return directory + "/" + file;
}

static bool fileExists(const std::string &path)
{
    std::ifstream probe(path, std::ios::binary);
    return static_cast<bool>(probe);
}

static std::string findFirstExisting(const std::vector<std::string> &candidates)
{
    for (const std::string &candidate : candidates) {
        if (fileExists(candidate)) return candidate;
    }
    return std::string();
}

static std::string findNearExecutable(const std::string &filename)
{
    std::vector<std::string> candidates;
    const std::string exe_dir = directoryOf(executablePath());
    if (!exe_dir.empty()) candidates.push_back(joinPath(exe_dir, filename));
    candidates.push_back(filename);
    return findFirstExisting(candidates);
}

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

// Fully connected layers resolved from the weight files. Layer order and
// names are taken from the files, so any MLP works without code changes.

struct LayerRef
{
    const FP32Tensor *fp32_weight;
    const FP32Tensor *fp32_bias;
    const AFPEncodedTensor *afp_weight;
    const AFPEncodedTensor *afp_bias;
    std::size_t rows;
    std::size_t columns;

    /*
        Prepared form of the AFP weight (decoded + converted to
        AFP::Product once at load, not once per image) and the decoded
        single-value bias. Filled after discovery.
    */
    std::unique_ptr<AFPPreparedMatrix> prepared_weight;
    AFP::Value bias_value;
    bool has_bias_value = false;
};

/*
    Discover the fully connected layer stack from the weight files. A layer
    is a 2-D "<name>.weight" tensor optionally paired with a 1-D
    "<name>.bias" tensor of matching output rows; layer order follows file
    order (the training framework's state dict order). Names, layer count
    and dimensions are arbitrary, so any MLP depth/width works without code
    changes. Inference applies ReLU between hidden layers, matching the
    training scripts in this repository.
*/

static std::vector<LayerRef> discoverLayers(
    const std::vector<FP32Tensor> &fp32_weights,
    const AFPModel &afp_model,
    std::size_t input_size)
{
    static const std::string weight_suffix = ".weight";

    std::vector<LayerRef> layers;

    for (const FP32Tensor &fp32_weight_tensor : fp32_weights) {
        const std::string &weight_name = fp32_weight_tensor.name;

        if (weight_name.size() > weight_suffix.size() &&
            weight_name.compare(weight_name.size() - weight_suffix.size(),
                                weight_suffix.size(), weight_suffix) == 0) {
            // Only fully connected (matrix) layers are supported; anything
            // else with a .weight tensor (conv, batchnorm, ...) cannot be
            // compared by this tool.
            if (fp32_weight_tensor.shape.size() != 2) {
                throw std::runtime_error("Unsupported (non fully connected) layer: " + weight_name);
            }
        } else {
            continue;
        }

        const std::string prefix = weight_name.substr(0, weight_name.size() - weight_suffix.size());
        const std::string bias_name = prefix + ".bias";

        const FP32Tensor *fp32_bias = nullptr;
        for (const FP32Tensor &tensor : fp32_weights) {
            if (tensor.name == bias_name) fp32_bias = &tensor;
        }

        LayerRef layer;
        layer.fp32_weight = &fp32_weight_tensor;
        layer.fp32_bias = fp32_bias;
        layer.rows = static_cast<std::size_t>(fp32_weight_tensor.shape[0]);
        layer.columns = static_cast<std::size_t>(fp32_weight_tensor.shape[1]);

        if (fp32_weight_tensor.values.size() != layer.rows * layer.columns) {
            throw std::runtime_error("FP32 weight size does not match shape for " + weight_name);
        }
        if (fp32_bias && fp32_bias->values.size() != layer.rows) {
            throw std::runtime_error("FP32 bias size does not match shape for " + bias_name);
        }

        const AFPModelTensor *afp_weight = nullptr;
        const AFPModelTensor *afp_bias = nullptr;
        for (const AFPModelTensor &tensor : afp_model.tensors) {
            if (tensor.name == weight_name) afp_weight = &tensor;
            if (tensor.name == bias_name) afp_bias = &tensor;
        }
        if (!afp_weight) {
            throw std::runtime_error("AFP model is missing " + weight_name);
        }
        if ((afp_bias != nullptr) != (fp32_bias != nullptr)) {
            throw std::runtime_error("Bias presence differs between the FP32 and AFP models for " + prefix);
        }
        if (afp_weight->tensor.size() != layer.rows * layer.columns) {
            throw std::runtime_error("AFP weight size does not match the FP32 model for " + weight_name);
        }
        if (afp_bias && afp_bias->tensor.size() != layer.rows) {
            throw std::runtime_error("AFP bias size does not match the FP32 model for " + bias_name);
        }

        layer.afp_weight = &afp_weight->tensor;
        layer.afp_bias = afp_bias ? &afp_bias->tensor : nullptr;

        /*
            Prepare the weight once at load: decode the bitstream and
            convert every weight to its AFP::Product form. Also decode
            the single-value bias. (The previous code re-parsed the
            whole weight bitstream for every image.)
        */
        layer.prepared_weight =
            AFPArithmetic::prepareMatrix(afp_weight->tensor, layer.rows, layer.columns);
        if (layer.afp_bias) {
            layer.bias_value = AFPArithmetic::decodeAFPValues(*layer.afp_bias).at(0);
            layer.has_bias_value = true;
        }

        layers.push_back(std::move(layer));
    }

    if (layers.empty()) {
        throw std::runtime_error("No 2-D *.weight layers found in the FP32 model.");
    }
    if (layers.front().columns != input_size) {
        throw std::runtime_error(
            "Model expects " + std::to_string(layers.front().columns) +
            " inputs but images provide " + std::to_string(input_size));
    }

    return layers;
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
    const std::vector<LayerRef> &layers,
    const std::vector<float> &input)
{
    std::vector<float> current = input;

    for (std::size_t l = 0; l < layers.size(); ++l) {
        const LayerRef &layer = layers[l];

        current = fp32MatrixVectorMultiply(layer.fp32_weight->values, current, layer.rows, layer.columns);

        if (layer.fp32_bias) {
            fp32AddBias(current, layer.fp32_bias->values);
        }

        if (l + 1 < layers.size()) {
            fp32ReLU(current);
        }
    }

    return current;
}

/*
    AFP inference over the prepared-weight + unpacked-value pipeline:
    weights are decoded once at load, activations stay as unpacked
    AFP::Value vectors between layers, and encoding happens only at the
    I/O boundary. Arithmetic is still AFP-native (integer
    Product/Accumulator) and per-layer rounding is identical to the
    packed pipeline (verified bit-exact in tests/test_afp_equivalence.cpp).
*/
static std::vector<AFP::Value> runAFPInferencePrepared(
    const std::vector<LayerRef> &layers,
    const std::vector<AFP::Value> &input)
{
    std::vector<AFP::Value> current = input;

    for (std::size_t l = 0; l < layers.size(); ++l) {
        const LayerRef &layer = layers[l];

        current = AFPArithmetic::matrixVectorMultiplyUnpacked(*layer.prepared_weight, current);

        if (layer.has_bias_value) {
            for (AFP::Value &v : current) v = v.add(layer.bias_value);
        }

        if (l + 1 < layers.size()) {
            current = AFPArithmetic::reluUnpacked(current);
        }
    }

    return current;
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
    const std::vector<LayerRef>* layers;
    AFPQuantizer* quantizer;
    std::atomic<std::size_t>* fp32_correct;
    std::atomic<std::size_t>* afp_correct;
    std::atomic<std::size_t>* agreement;
    bool score_against_labels;
    std::atomic<double>* cumulative_mae;
    std::atomic<double>* cumulative_rmse;
    std::atomic<double>* cumulative_max_error;
    std::atomic<std::uint64_t>* fp32_elapsed_us;
    std::atomic<std::uint64_t>* afp_encode_elapsed_us;
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
    for (std::size_t i = task.start_idx; i < task.end_idx; ++i) {
        const std::vector<float> &image = (*task.images)[i];
        
        // FP32 inference
        auto fp32_start = std::chrono::high_resolution_clock::now();
        std::vector<float> fp32_output = runFP32Inference(*task.layers, image);
        auto fp32_end = std::chrono::high_resolution_clock::now();
        
        std::size_t fp32_pred = argmax(fp32_output);
        
        task.fp32_elapsed_us->fetch_add(
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(fp32_end - fp32_start).count()));

        // AFP inference (prepared weights, unpacked activations)
        auto afp_encode_start = std::chrono::high_resolution_clock::now();
        AFPEncodedTensor afp_input = task.quantizer->encode(image);
        auto afp_encode_end = std::chrono::high_resolution_clock::now();
        auto afp_start = std::chrono::high_resolution_clock::now();
        std::vector<AFP::Value> afp_output_values =
            runAFPInferencePrepared(*task.layers, AFPArithmetic::decodeAFPValues(afp_input));
        auto afp_end = std::chrono::high_resolution_clock::now();
        
        std::vector<float> afp_decoded = task.quantizer->decode(
            AFPArithmetic::encodeAFPValues(afp_output_values, task.quantizer->getConfig()));
        std::size_t afp_pred = argmax(afp_decoded);
        
        if (fp32_pred == afp_pred) {
            task.agreement->fetch_add(1);
        }

        // Label accuracy is only meaningful when the network output matches
        // the 10 MNIST classes.
        if (task.score_against_labels) {
            const std::size_t truth = static_cast<std::size_t>((*task.labels)[i]);

            if (fp32_pred == truth) {
                task.fp32_correct->fetch_add(1);
            }
            if (afp_pred == truth) {
                task.afp_correct->fetch_add(1);
            }
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
        task.afp_encode_elapsed_us->fetch_add(
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(afp_encode_end - afp_encode_start).count()));
    }
}

static void printUsage(const char *executable)
{
    std::cout
        << "Usage: " << executable << " [count] [options]\n"
        << "  --count N    Number of MNIST test images to evaluate (default: 10000)\n"
        << "  --fp32 PATH  FP32 weight file (default: mnist_mlp_weights.bin next to the executable)\n"
        << "  --afp PATH   AFP weight file (default: <FP32 stem>_afp.bin next to the FP32 file)\n"
        << "  --data DIR   Directory holding the MNIST IDX test files\n";
}

/*
    Locate the directory holding the MNIST IDX test files: the --data
    override wins, otherwise walk upwards from the current directory and
    from the executable's directory looking for a data/MNIST/raw directory
    containing both test files.
*/

static std::string findMNISTDirectory(const std::string &override_dir)
{
    if (!override_dir.empty()) return override_dir;

    std::vector<std::string> bases;
    bases.push_back(std::string("."));

    const std::string exe_dir = directoryOf(executablePath());
    if (!exe_dir.empty()) bases.push_back(exe_dir);

    std::vector<std::string> candidates;
    for (const std::string &base : bases) {
        std::string dir = base;
        for (int depth = 0; depth < 6 && !dir.empty(); ++depth) {
            candidates.push_back(joinPath(dir, "data/MNIST/raw"));
            const std::string parent = directoryOf(dir);
            if (parent == dir) break;
            dir = parent;
        }
    }

    const std::vector<std::string> required = {
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    };

    for (const std::string &candidate : candidates) {
        bool complete = true;
        for (const std::string &file : required) {
            if (!fileExists(joinPath(candidate, file))) {
                complete = false;
                break;
            }
        }
        if (complete) return candidate;
    }

    return std::string();
}

int main(int argc, char *argv[])
{
    try {
        std::size_t test_count = 10000;
        std::string fp32_override;
        std::string afp_override;
        std::string data_override;

        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];

            if (argument == "--count" && i + 1 < argc) {
                test_count = std::stoul(argv[++i]);
            } else if (argument == "--fp32" && i + 1 < argc) {
                fp32_override = argv[++i];
            } else if (argument == "--afp" && i + 1 < argc) {
                afp_override = argv[++i];
            } else if (argument == "--data" && i + 1 < argc) {
                data_override = argv[++i];
            } else if (argument == "--help" || argument == "-h") {
                printUsage(argv[0]);
                return 0;
            } else if (i == 1) {
                test_count = std::stoul(argument); // bare image count for convenience
            } else {
                std::cerr << "Unknown argument: " << argument << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }

        std::string fp32_filename = fp32_override;
        if (fp32_filename.empty()) {
            fp32_filename = findNearExecutable("mnist_mlp_weights.bin");
        }
        if (fp32_filename.empty()) {
            std::cerr << "FP32 weight file not found (looked for mnist_mlp_weights.bin"
                      << " next to the executable and in the current directory).\n";
            printUsage(argv[0]);
            return 1;
        }

        std::string afp_filename = afp_override;
        if (afp_filename.empty()) {
            afp_filename = joinPath(directoryOf(fp32_filename), stemOf(fp32_filename) + "_afp.bin");
            if (!fileExists(afp_filename)) {
                afp_filename = findNearExecutable(stemOf(fp32_filename) + "_afp.bin");
            }
        }
        if (afp_filename.empty() || !fileExists(afp_filename)) {
            std::cerr << "AFP weight file not found for " << fp32_filename << "\n";
            printUsage(argv[0]);
            return 1;
        }

        const std::string mnist_dir = findMNISTDirectory(data_override);
        if (mnist_dir.empty()) {
            std::cerr << "MNIST test data not found (looked for data/MNIST/raw above"
                      << " the current directory and above the executable's directory).\n";
            printUsage(argv[0]);
            return 1;
        }

        const std::string images_filename = joinPath(mnist_dir, "t10k-images-idx3-ubyte");
        const std::string labels_filename = joinPath(mnist_dir, "t10k-labels-idx1-ubyte");

        std::cout << "FP32 weights: " << fp32_filename << "\n";
        std::cout << "AFP weights:  " << afp_filename << "\n";
        std::cout << "MNIST data:   " << mnist_dir << "\n\n";

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

        const std::size_t input_size = mnist.images.empty() ? 0 : mnist.images.front().size();
        const std::vector<LayerRef> layers = discoverLayers(fp32_weights, afp_model, input_size);

        // Label accuracy is only well-defined for the 10 MNIST classes;
        // otherwise the FP32/AFP prediction agreement is reported instead.
        const bool score_against_labels = layers.back().rows == 10;

        std::cout << "Discovered " << layers.size() << " fully connected layers:";
        for (const LayerRef &layer : layers) {
            std::cout << " " << layer.rows << "x" << layer.columns;
        }
        std::cout << "\n\n";

        AFPQuantizer quantizer(afp_model.config);

        // Metrics
        std::atomic<std::size_t> fp32_correct(0);
        std::atomic<std::size_t> afp_correct(0);
        std::atomic<std::size_t> agreement(0);
        std::atomic<double> cumulative_mae(0.0);
        std::atomic<double> cumulative_rmse(0.0);
        std::atomic<double> cumulative_max_error(0.0);
        std::atomic<std::uint64_t> fp32_elapsed_us(0);
        std::atomic<std::uint64_t> afp_encode_elapsed_us(0);
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
            task.layers = &layers;
            task.quantizer = &quantizer;
            task.fp32_correct = &fp32_correct;
            task.afp_correct = &afp_correct;
            task.agreement = &agreement;
            task.score_against_labels = score_against_labels;
            task.cumulative_mae = &cumulative_mae;
            task.cumulative_rmse = &cumulative_rmse;
            task.cumulative_max_error = &cumulative_max_error;
            task.fp32_elapsed_us = &fp32_elapsed_us;
            task.afp_encode_elapsed_us = &afp_encode_elapsed_us;
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
            std::vector<float> fp32_output = runFP32Inference(layers, image);
            auto fp32_end = std::chrono::high_resolution_clock::now();
            
            std::size_t fp32_pred = argmax(fp32_output);
            
            // AFP input encoding
            auto afp_encode_start = std::chrono::high_resolution_clock::now();
            AFPEncodedTensor afp_input = quantizer.encode(image);
            auto afp_encode_end = std::chrono::high_resolution_clock::now();

            // AFP inference (prepared weights, unpacked activations)
            auto afp_start = std::chrono::high_resolution_clock::now();
            std::vector<AFP::Value> afp_output_values =
                runAFPInferencePrepared(layers, AFPArithmetic::decodeAFPValues(afp_input));
            auto afp_end = std::chrono::high_resolution_clock::now();
            
            std::vector<float> afp_decoded = quantizer.decode(
                AFPArithmetic::encodeAFPValues(afp_output_values, quantizer.getConfig()));
            std::size_t afp_pred = argmax(afp_decoded);
            
            if (fp32_pred == afp_pred) {
                agreement++;
            }

            if (score_against_labels) {
                if (fp32_pred == static_cast<std::size_t>(mnist.labels[i])) {
                    fp32_correct++;
                }
                if (afp_pred == static_cast<std::size_t>(mnist.labels[i])) {
                    afp_correct++;
                }
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
            afp_encode_elapsed_us.fetch_add(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(afp_encode_end - afp_encode_start).count()));
        }
        #endif

        // Calculate final metrics
        double fp32_accuracy = static_cast<double>(fp32_correct) / static_cast<double>(count) * 100.0;
        double afp_accuracy = static_cast<double>(afp_correct) / static_cast<double>(count) * 100.0;
        const double agreement_pct = static_cast<double>(agreement.load()) / static_cast<double>(count) * 100.0;
        double final_mae = cumulative_mae / static_cast<double>(count);
        double final_rmse = cumulative_rmse / static_cast<double>(count);
        double final_max_error = cumulative_max_error / static_cast<double>(count);

        std::cout << "\n========================================\n";
        std::cout << "INFERENCE RESULTS\n";
        std::cout << "========================================\n";
        if (score_against_labels) {
            std::cout << "FP32 Accuracy: " << std::fixed << std::setprecision(2) << fp32_accuracy << "%\n";
            std::cout << "AFP Accuracy: " << std::fixed << std::setprecision(2) << afp_accuracy << "%\n";
            std::cout << "Accuracy Drop: " << std::fixed << std::setprecision(2) << (fp32_accuracy - afp_accuracy) << "%\n";
        } else {
            std::cout << "Output layer produces " << layers.back().rows
                      << " values (not the 10 MNIST classes); reporting FP32/AFP agreement.\n";
            std::cout << "FP32/AFP Agreement: " << std::fixed << std::setprecision(2) << agreement_pct << "%\n";
        }
        std::cout << "MAE: " << std::scientific << std::setprecision(4) << final_mae << '\n';
        std::cout << "RMSE: " << std::scientific << std::setprecision(4) << final_rmse << '\n';
        std::cout << "Max Error: " << std::scientific << std::setprecision(4) << final_max_error << '\n';
        std::cout << "========================================\n";

        const double count_double = static_cast<double>(count);
        const double fp32_seconds = static_cast<double>(fp32_elapsed_us.load()) / 1e6;
        const double afp_encode_seconds = static_cast<double>(afp_encode_elapsed_us.load()) / 1e6;
        const double afp_seconds = static_cast<double>(afp_elapsed_us.load()) / 1e6;
        const double delta_seconds = afp_seconds - fp32_seconds;
        const double fp32_us_per_image = fp32_seconds * 1e6 / count_double;
        const double afp_encode_us_per_image = afp_encode_seconds * 1e6 / count_double;
        const double afp_us_per_image = afp_seconds * 1e6 / count_double;
        const double delta_us_per_image = afp_us_per_image - fp32_us_per_image;
        const double speedup = (afp_us_per_image > 0.0) ? (fp32_us_per_image / afp_us_per_image) : 0.0;

        std::cout << "\n========================================\n";
        std::cout << "TIME SUMMARY\n";
        std::cout << "========================================\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "FP32 total: " << fp32_seconds << " s (" << fp32_us_per_image << " us/image)\n";
        std::cout << "AFP encode: " << afp_encode_seconds << " s (" << afp_encode_us_per_image << " us/image)\n";
        std::cout << "AFP inference: " << afp_seconds << " s (" << afp_us_per_image << " us/image)\n";
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