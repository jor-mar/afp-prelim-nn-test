#include "../../include/afp_encoded_tensor.hpp"
#include "../../include/afp_math.hpp"

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

static std::uint32_t readUint32(
    std::ifstream &file)
{
    std::uint32_t value = 0;

    file.read(
        reinterpret_cast<char *>(&value),
        sizeof(value));

    if (!file)
    {
        throw std::runtime_error(
            "Failed to read uint32.");
    }

    return value;
}

static std::uint64_t readUint64(
    std::ifstream &file)
{
    std::uint64_t value = 0;

    file.read(
        reinterpret_cast<char *>(&value),
        sizeof(value));

    if (!file)
    {
        throw std::runtime_error(
            "Failed to read uint64.");
    }

    return value;
}

static std::uint8_t readUint8(
    std::ifstream &file)
{
    std::uint8_t value = 0;

    file.read(
        reinterpret_cast<char *>(&value),
        sizeof(value));

    if (!file)
    {
        throw std::runtime_error(
            "Failed to read uint8.");
    }

    return value;
}

static float readFloat(
    std::ifstream &file)
{
    float value = 0.0f;

    file.read(
        reinterpret_cast<char *>(&value),
        sizeof(value));

    if (!file)
    {
        throw std::runtime_error(
            "Failed to read float.");
    }

    return value;
}

static std::uint32_t readBigEndianUint32(
    std::ifstream &file)
{
    std::uint8_t bytes[4];

    file.read(
        reinterpret_cast<char *>(bytes),
        4);

    if (!file)
    {
        throw std::runtime_error(
            "Failed to read big-endian uint32.");
    }

    return
        (static_cast<std::uint32_t>(bytes[0]) << 24) |
        (static_cast<std::uint32_t>(bytes[1]) << 16) |
        (static_cast<std::uint32_t>(bytes[2]) << 8) |
        static_cast<std::uint32_t>(bytes[3]);
}

static void checkFile(
    const std::ifstream &file,
    const std::string &filename)
{
    if (!file)
    {
        throw std::runtime_error(
            "Failed to open file: " + filename);
    }
}

static std::vector<FP32Tensor> loadFP32Model(
    const std::string &filename)
{
    std::ifstream input(
        filename,
        std::ios::binary);

    checkFile(input, filename);

    std::uint32_t tensor_count =
        readUint32(input);

    std::vector<FP32Tensor> tensors;

    tensors.reserve(tensor_count);

    for (std::uint32_t i = 0; i < tensor_count; ++i)
    {
        FP32Tensor tensor;

        std::uint32_t name_length =
            readUint32(input);

        tensor.name.resize(name_length);

        input.read(
            tensor.name.data(),
            static_cast<std::streamsize>(
                name_length));

        if (!input)
        {
            throw std::runtime_error(
                "Failed to read FP32 tensor name.");
        }

        std::uint32_t dimension_count =
            readUint32(input);

        tensor.shape.resize(
            dimension_count);

        for (std::uint32_t d = 0;
             d < dimension_count;
             ++d)
        {
            tensor.shape[d] =
                readUint64(input);
        }

        std::uint64_t value_count =
            readUint64(input);

        tensor.values.resize(
            static_cast<std::size_t>(
                value_count));

        for (std::uint64_t j = 0;
             j < value_count;
             ++j)
        {
            tensor.values[
                static_cast<std::size_t>(j)] =
                readFloat(input);
        }

        tensors.push_back(
            std::move(tensor));
    }

    return tensors;
}

static AFPModel loadAFPModel(
    const std::string &filename)
{
    std::ifstream input(
        filename,
        std::ios::binary);

    checkFile(input, filename);

    char magic[4];

    input.read(
        magic,
        4);

    if (!input)
    {
        throw std::runtime_error(
            "Failed to read AFP magic.");
    }

    if (magic[0] != 'A' ||
        magic[1] != 'F' ||
        magic[2] != 'P' ||
        magic[3] != '8')
    {
        throw std::runtime_error(
            "Invalid AFP model magic.");
    }

    std::uint32_t version =
        readUint32(input);

    if (version != 2)
    {
        throw std::runtime_error(
            "Unsupported AFP model version: " +
            std::to_string(version));
    }

    std::uint32_t tensor_count =
        readUint32(input);

    AFPModel model;

    model.config.block_size =
        static_cast<std::size_t>(
            readUint64(input));

    model.config.exponent_bits =
        static_cast<int>(
            readUint32(input));

    model.config.characterization_bits =
        static_cast<int>(
            readUint32(input));

    model.config.offset_bits =
        static_cast<int>(
            readUint32(input));

    model.config.mantissa_bits =
        static_cast<int>(
            readUint32(input));

    model.config.enable_positive_fields =
        readUint8(input) != 0;

    model.config.enable_zero_fields =
        readUint8(input) != 0;

    AFPQuantizer quantizer(model.config);

    model.tensors.reserve(tensor_count);

    for (std::uint32_t i = 0;
         i < tensor_count;
         ++i)
    {
        AFPModelTensor model_tensor;

        std::uint32_t name_length =
            readUint32(input);

        model_tensor.name.resize(
            name_length);

        input.read(
            model_tensor.name.data(),
            static_cast<std::streamsize>(
                name_length));

        if (!input)
        {
            throw std::runtime_error(
                "Failed to read AFP tensor name.");
        }

        std::uint32_t dimension_count =
            readUint32(input);

        model_tensor.shape.resize(
            dimension_count);

        for (std::uint32_t d = 0;
             d < dimension_count;
             ++d)
        {
            model_tensor.shape[d] =
                readUint64(input);
        }

        std::uint64_t value_count =
            readUint64(input);

        std::uint64_t bit_count =
            readUint64(input);

        std::uint64_t byte_count =
            readUint64(input);

        std::uint64_t block_count =
            readUint64(input);

        std::vector<std::size_t> block_offsets;

        block_offsets.reserve(
            static_cast<std::size_t>(
                block_count));

        for (std::uint64_t b = 0;
             b < block_count;
             ++b)
        {
            block_offsets.push_back(
                static_cast<std::size_t>(
                    readUint64(input)));
        }

        std::vector<std::uint8_t> data(
            static_cast<std::size_t>(
                byte_count));

        if (!data.empty())
        {
            input.read(
                reinterpret_cast<char *>(
                    data.data()),
                static_cast<std::streamsize>(
                    data.size()));

            if (!input)
            {
                throw std::runtime_error(
                    "Failed to read AFP tensor data.");
            }
        }

        model_tensor.tensor =
            quantizer.load(
                data,
                static_cast<std::size_t>(
                    bit_count),
                static_cast<std::size_t>(
                    value_count),
                model.config,
                block_offsets);

        model.tensors.push_back(
            std::move(model_tensor));
    }

    return model;
}

static const FP32Tensor &findFP32Tensor(
    const std::vector<FP32Tensor> &tensors,
    const std::string &name)
{
    for (const FP32Tensor &tensor : tensors)
    {
        if (tensor.name == name)
        {
            return tensor;
        }
    }

    throw std::runtime_error(
        "FP32 tensor not found: " + name);
}

static const AFPEncodedTensor &findAFPTensor(
    const AFPModel &model,
    const std::string &name)
{
    for (const AFPModelTensor &tensor :
         model.tensors)
    {
        if (tensor.name == name)
        {
            return tensor.tensor;
        }
    }

    throw std::runtime_error(
        "AFP tensor not found: " + name);
}

static void readIDXImages(
    const std::string &filename,
    std::vector<std::vector<float>> &images)
{
    std::ifstream input(
        filename,
        std::ios::binary);

    checkFile(input, filename);

    std::uint32_t magic =
        readBigEndianUint32(input);

    std::uint32_t image_count =
        readBigEndianUint32(input);

    std::uint32_t rows =
        readBigEndianUint32(input);

    std::uint32_t columns =
        readBigEndianUint32(input);

    if (magic != 2051)
    {
        throw std::runtime_error(
            "Invalid MNIST image file.");
    }

    images.resize(image_count);

    const std::size_t pixels =
        static_cast<std::size_t>(rows) *
        static_cast<std::size_t>(columns);

    for (std::uint32_t i = 0;
         i < image_count;
         ++i)
    {
        images[i].resize(pixels);

        for (std::size_t j = 0;
             j < pixels;
             ++j)
        {
            std::uint8_t pixel = 0;

            input.read(
                reinterpret_cast<char *>(&pixel),
                1);

            if (!input)
            {
                throw std::runtime_error(
                    "Failed to read MNIST image.");
            }

            /*
                Same normalization used during training:

                    (pixel / 255 - 0.1307) / 0.3081
            */

            float normalized =
                static_cast<float>(pixel) / 255.0f;

            images[i][j] =
                (normalized - 0.1307f) / 0.3081f;
        }
    }
}

static void readIDXLabels(
    const std::string &filename,
    std::vector<std::uint8_t> &labels)
{
    std::ifstream input(
        filename,
        std::ios::binary);

    checkFile(input, filename);

    std::uint32_t magic =
        readBigEndianUint32(input);

    std::uint32_t label_count =
        readBigEndianUint32(input);

    if (magic != 2049)
    {
        throw std::runtime_error(
            "Invalid MNIST label file.");
    }

    labels.resize(label_count);

    for (std::uint32_t i = 0;
         i < label_count;
         ++i)
    {
        input.read(
            reinterpret_cast<char *>(&labels[i]),
            1);

        if (!input)
        {
            throw std::runtime_error(
                "Failed to read MNIST label.");
        }
    }
}

static std::vector<float> fp32MatrixVectorMultiply(
    const std::vector<float> &weights,
    const std::vector<float> &input,
    std::size_t rows,
    std::size_t columns)
{
    if (weights.size() != rows * columns)
    {
        throw std::runtime_error(
            "FP32 weight size does not match matrix dimensions.");
    }

    if (input.size() != columns)
    {
        throw std::runtime_error(
            "FP32 input size does not match matrix dimensions.");
    }

    std::vector<float> output(rows, 0.0f);

    for (std::size_t row = 0;
         row < rows;
         ++row)
    {
        float sum = 0.0f;

        for (std::size_t column = 0;
             column < columns;
             ++column)
        {
            sum +=
                weights[row * columns + column] *
                input[column];
        }

        output[row] = sum;
    }

    return output;
}

static void fp32AddBias(
    std::vector<float> &values,
    const std::vector<float> &bias)
{
    if (values.size() != bias.size())
    {
        throw std::runtime_error(
            "Bias size does not match output size.");
    }

    for (std::size_t i = 0;
         i < values.size();
         ++i)
    {
        values[i] += bias[i];
    }
}

static void fp32ReLU(
    std::vector<float> &values)
{
    for (float &value : values)
    {
        if (value < 0.0f)
        {
            value = 0.0f;
        }
    }
}

static std::size_t argmax(
    const std::vector<float> &values)
{
    if (values.empty())
    {
        throw std::runtime_error(
            "Cannot calculate argmax of empty vector.");
    }

    std::size_t index = 0;

    for (std::size_t i = 1;
         i < values.size();
         ++i)
    {
        if (values[i] > values[index])
        {
            index = i;
        }
    }

    return index;
}

static std::vector<float> runFP32Inference(
    const std::vector<FP32Tensor> &weights,
    const std::vector<float> &input)
{
    const FP32Tensor &fc1_weight =
        findFP32Tensor(
            weights,
            "fc1.weight");

    const FP32Tensor &fc1_bias =
        findFP32Tensor(
            weights,
            "fc1.bias");

    const FP32Tensor &fc2_weight =
        findFP32Tensor(
            weights,
            "fc2.weight");

    const FP32Tensor &fc2_bias =
        findFP32Tensor(
            weights,
            "fc2.bias");

    const FP32Tensor &fc3_weight =
        findFP32Tensor(
            weights,
            "fc3.weight");

    const FP32Tensor &fc3_bias =
        findFP32Tensor(
            weights,
            "fc3.bias");

    std::vector<float> layer1 =
        fp32MatrixVectorMultiply(
            fc1_weight.values,
            input,
            128,
            784);

    fp32AddBias(
        layer1,
        fc1_bias.values);

    fp32ReLU(layer1);

    std::vector<float> layer2 =
        fp32MatrixVectorMultiply(
            fc2_weight.values,
            layer1,
            64,
            128);

    fp32AddBias(
        layer2,
        fc2_bias.values);

    fp32ReLU(layer2);

    std::vector<float> layer3 =
        fp32MatrixVectorMultiply(
            fc3_weight.values,
            layer2,
            10,
            64);

    fp32AddBias(
        layer3,
        fc3_bias.values);

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
    /*
        Everything after the input has been encoded
        remains AFP.

        No AFP tensor is decoded to float between
        neural-network layers.
    */

    AFPEncodedTensor layer1 =
        AFPArithmetic::matrixVectorMultiply(
            fc1_weight,
            input,
            128,
            784);

    layer1 =
        AFPArithmetic::add(
            layer1,
            fc1_bias);

    layer1 =
        AFPArithmetic::relu(
            layer1);

    AFPEncodedTensor layer2 =
        AFPArithmetic::matrixVectorMultiply(
            fc2_weight,
            layer1,
            64,
            128);

    layer2 =
        AFPArithmetic::add(
            layer2,
            fc2_bias);

    layer2 =
        AFPArithmetic::relu(
            layer2);

    AFPEncodedTensor layer3 =
        AFPArithmetic::matrixVectorMultiply(
            fc3_weight,
            layer2,
            10,
            64);

    layer3 =
        AFPArithmetic::add(
            layer3,
            fc3_bias);

    return layer3;
}

static double calculateMAE(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error(
            "MAE vectors have different sizes.");
    }

    if (a.empty())
    {
        return 0.0;
    }

    double total = 0.0;

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        total +=
            std::abs(
                static_cast<double>(a[i]) -
                static_cast<double>(b[i]));
    }

    return total /
           static_cast<double>(a.size());
}

static double calculateRMSE(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error(
            "RMSE vectors have different sizes.");
    }

    if (a.empty())
    {
        return 0.0;
    }

    double total = 0.0;

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        double difference =
            static_cast<double>(a[i]) -
            static_cast<double>(b[i]);

        total +=
            difference * difference;
    }

    return std::sqrt(
        total /
        static_cast<double>(a.size()));
}

static double calculateMaxError(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error(
            "Max-error vectors have different sizes.");
    }

    double maximum = 0.0;

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        double difference =
            std::abs(
                static_cast<double>(a[i]) -
                static_cast<double>(b[i]));

        if (difference > maximum)
        {
            maximum = difference;
        }
    }

    return maximum;
}

int main()
{
    try
    {
        const std::string fp32_filename =
            "experiments/mnist_mlp_1/mnist_mlp_weights.bin";

        const std::string afp_filename =
            "experiments/mnist_mlp_1/mnist_mlp_weights_afp.bin";

        const std::string images_filename =
            "data/MNIST/raw/t10k-images-idx3-ubyte";

        const std::string labels_filename =
            "data/MNIST/raw/t10k-labels-idx1-ubyte";

        /*
            Number of images to test.

            Set this to 10000 for the entire MNIST
            test set.
        */

        const std::size_t test_count = 10000;

        std::cout
            << "Loading FP32 model...\n";

        std::vector<FP32Tensor> fp32_weights =
            loadFP32Model(
                fp32_filename);

        std::cout
            << "Loading AFP model...\n";

        AFPModel afp_model =
            loadAFPModel(
                afp_filename);

        std::cout
            << "Loading MNIST test images...\n";

        MNISTData mnist;

        readIDXImages(
            images_filename,
            mnist.images);

        readIDXLabels(
            labels_filename,
            mnist.labels);

        std::size_t count =
            std::min(
                test_count,
                std::min(
                    mnist.images.size(),
                    mnist.labels.size()));

        if (count == 0)
        {
            throw std::runtime_error(
                "No MNIST test images found.");
        }

        std::cout
            << "\nLoaded "
            << count
            << " test images.\n\n";

        /*
            Retrieve all six AFP model tensors.
        */

        const AFPEncodedTensor &afp_fc1_weight =
            findAFPTensor(
                afp_model,
                "fc1.weight");

        const AFPEncodedTensor &afp_fc1_bias =
            findAFPTensor(
                afp_model,
                "fc1.bias");

        const AFPEncodedTensor &afp_fc2_weight =
            findAFPTensor(
                afp_model,
                "fc2.weight");

        const AFPEncodedTensor &afp_fc2_bias =
            findAFPTensor(
                afp_model,
                "fc2.bias");

        const AFPEncodedTensor &afp_fc3_weight =
            findAFPTensor(
                afp_model,
                "fc3.weight");

        const AFPEncodedTensor &afp_fc3_bias =
            findAFPTensor(
                afp_model,
                "fc3.bias");

        AFPQuantizer quantizer(
            afp_model.config);

        /*
            Metrics.
        */

        std::size_t fp32_correct = 0;
        std::size_t afp_correct = 0;
        std::size_t prediction_agreements = 0;

        double total_mae = 0.0;
        double total_rmse = 0.0;
        double maximum_error = 0.0;

        /*
            Timing.

            FP32 time:
                actual FP32 inference only.

            AFP inference time:
                AFP operations only, after input encoding.

            AFP encoding time:
                converting FP32 MNIST input to AFP.

            AFP end-to-end:
                input encoding + AFP inference.
        */

        std::chrono::nanoseconds fp32_total_time(0);
        std::chrono::nanoseconds afp_encode_total_time(0);
        std::chrono::nanoseconds afp_inference_total_time(0);

        std::cout
            << "Running inference...\n";

        for (std::size_t image_index = 0;
             image_index < count;
             ++image_index)
        {
            const std::vector<float> &input =
                mnist.images[image_index];

            /*
                FP32 inference.
            */

            auto fp32_start =
                std::chrono::steady_clock::now();

            std::vector<float> fp32_output =
                runFP32Inference(
                    fp32_weights,
                    input);

            auto fp32_end =
                std::chrono::steady_clock::now();

            fp32_total_time +=
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    fp32_end - fp32_start);

            /*
                AFP input encoding.

                This is outside AFP inference timing
                because the neural-network computation
                itself is entirely AFP.
            */

            auto encode_start =
                std::chrono::steady_clock::now();

            AFPEncodedTensor afp_input =
                quantizer.encode(input);

            auto encode_end =
                std::chrono::steady_clock::now();

            afp_encode_total_time +=
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    encode_end - encode_start);

            /*
                Fully AFP neural-network inference.
            */

            auto afp_start =
                std::chrono::steady_clock::now();

            AFPEncodedTensor afp_output =
                runAFPInference(
                    afp_fc1_weight,
                    afp_fc1_bias,
                    afp_fc2_weight,
                    afp_fc2_bias,
                    afp_fc3_weight,
                    afp_fc3_bias,
                    afp_input);

            auto afp_end =
                std::chrono::steady_clock::now();

            afp_inference_total_time +=
                std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    afp_end - afp_start);

            /*
                Decode only the final AFP output.

                This does NOT participate in AFP inference.
                It is only needed to compare the AFP result
                against the FP32 result.
            */

            std::vector<float> afp_output_decoded =
                quantizer.decode(
                    afp_output);

            /*
                Accuracy.
            */

            std::size_t fp32_prediction =
                argmax(fp32_output);

            std::size_t afp_prediction =
                argmax(afp_output_decoded);

            std::size_t truth =
                static_cast<std::size_t>(
                    mnist.labels[image_index]);

            if (fp32_prediction == truth)
            {
                ++fp32_correct;
            }

            if (afp_prediction == truth)
            {
                ++afp_correct;
            }

            if (fp32_prediction ==
                afp_prediction)
            {
                ++prediction_agreements;
            }

            /*
                Numerical similarity.
            */

            double mae =
                calculateMAE(
                    fp32_output,
                    afp_output_decoded);

            double rmse =
                calculateRMSE(
                    fp32_output,
                    afp_output_decoded);

            double max_error =
                calculateMaxError(
                    fp32_output,
                    afp_output_decoded);

            total_mae += mae;
            total_rmse += rmse;

            if (max_error > maximum_error)
            {
                maximum_error = max_error;
            }

            /*
                Progress.
            */

            if ((image_index + 1) % 1000 == 0 ||
                image_index + 1 == count)
            {
                std::cout
                    << "\rProcessed "
                    << (image_index + 1)
                    << " / "
                    << count
                    << std::flush;
            }
        }

        std::cout
            << "\n\n";

        /*
            Calculate timing statistics.
        */

        double fp32_total_ms =
            static_cast<double>(
                fp32_total_time.count()) /
            1'000'000.0;

        double afp_encode_total_ms =
            static_cast<double>(
                afp_encode_total_time.count()) /
            1'000'000.0;

        double afp_inference_total_ms =
            static_cast<double>(
                afp_inference_total_time.count()) /
            1'000'000.0;

        double afp_end_to_end_total_ms =
            afp_encode_total_ms +
            afp_inference_total_ms;

        double fp32_average_ms =
            fp32_total_ms /
            static_cast<double>(count);

        double afp_encode_average_ms =
            afp_encode_total_ms /
            static_cast<double>(count);

        double afp_inference_average_ms =
            afp_inference_total_ms /
            static_cast<double>(count);

        double afp_end_to_end_average_ms =
            afp_end_to_end_total_ms /
            static_cast<double>(count);

        double fp32_accuracy =
            100.0 *
            static_cast<double>(fp32_correct) /
            static_cast<double>(count);

        double afp_accuracy =
            100.0 *
            static_cast<double>(afp_correct) /
            static_cast<double>(count);

        double prediction_agreement =
            100.0 *
            static_cast<double>(
                prediction_agreements) /
            static_cast<double>(count);

        double average_mae =
            total_mae /
            static_cast<double>(count);

        double average_rmse =
            total_rmse /
            static_cast<double>(count);

        /*
            Final report.
        */

        std::cout
            << "========================================\n";

        std::cout
            << "           AFP vs FP32 MNIST\n";

        std::cout
            << "========================================\n\n";

        std::cout
            << "Images tested: "
            << count
            << "\n\n";

        std::cout
            << std::fixed
            << std::setprecision(4);

        std::cout
            << "TIMING\n"
            << "----------------------------------------\n";

        std::cout
            << "FP32 total:                 "
            << fp32_total_ms
            << " ms\n";

        std::cout
            << "FP32 average/image:        "
            << fp32_average_ms
            << " ms\n\n";

        std::cout
            << "AFP input encoding total:   "
            << afp_encode_total_ms
            << " ms\n";

        std::cout
            << "AFP input encoding/image:   "
            << afp_encode_average_ms
            << " ms\n\n";

        std::cout
            << "AFP inference total:        "
            << afp_inference_total_ms
            << " ms\n";

        std::cout
            << "AFP inference/image:        "
            << afp_inference_average_ms
            << " ms\n\n";

        std::cout
            << "AFP end-to-end total:       "
            << afp_end_to_end_total_ms
            << " ms\n";

        std::cout
            << "AFP end-to-end/image:       "
            << afp_end_to_end_average_ms
            << " ms\n\n";

        if (afp_inference_total_ms > 0.0)
        {
            std::cout
                << "FP32 / AFP inference speed: "
                << fp32_total_ms /
                       afp_inference_total_ms
                << "x\n";
        }

        if (afp_end_to_end_total_ms > 0.0)
        {
            std::cout
                << "FP32 / AFP end-to-end speed: "
                << fp32_total_ms /
                       afp_end_to_end_total_ms
                << "x\n";
        }

        std::cout
            << "\n";

        std::cout
            << "NUMERICAL ACCURACY\n"
            << "----------------------------------------\n";

        std::cout
            << "Logit MAE:                  "
            << average_mae
            << "\n";

        std::cout
            << "Logit RMSE:                 "
            << average_rmse
            << "\n";

        std::cout
            << "Maximum logit error:        "
            << maximum_error
            << "\n";

        std::cout
            << "Prediction agreement:       "
            << prediction_agreement
            << "%\n";

        std::cout
            << "\n";

        std::cout
            << "CLASSIFICATION ACCURACY\n"
            << "----------------------------------------\n";

        std::cout
            << "FP32 correct:               "
            << fp32_correct
            << " / "
            << count
            << "\n";

        std::cout
            << "FP32 accuracy:              "
            << fp32_accuracy
            << "%\n\n";

        std::cout
            << "AFP correct:                "
            << afp_correct
            << " / "
            << count
            << "\n";

        std::cout
            << "AFP accuracy:               "
            << afp_accuracy
            << "%\n\n";

        std::cout
            << "Accuracy difference:        "
            << afp_accuracy -
                   fp32_accuracy
            << " percentage points\n";

        std::cout
            << "\n========================================\n";

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "\nERROR: "
            << error.what()
            << '\n';

        return 1;
    }
}