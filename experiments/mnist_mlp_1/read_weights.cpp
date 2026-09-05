#include "../../include/afp_math.hpp"
#include "../../include/afp_encoded_tensor.hpp"
#include "../../include/encoded_tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Tensor
{
    std::string name;
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

struct MNISTDataset
{
    std::vector<std::vector<float>> images;
    std::vector<std::uint8_t> labels;
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

std::uint32_t readUInt32BE(std::ifstream &file)
{
    std::uint8_t bytes[4];

    file.read(
        reinterpret_cast<char *>(bytes),
        4);

    if (!file)
        throw std::runtime_error(
            "Failed to read big-endian uint32");

    return
        (static_cast<std::uint32_t>(bytes[0]) << 24) |
        (static_cast<std::uint32_t>(bytes[1]) << 16) |
        (static_cast<std::uint32_t>(bytes[2]) << 8) |
        static_cast<std::uint32_t>(bytes[3]);
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
    const std::uint32_t length =
        readUInt32(file);

    std::string value(
        length,
        '\0');

    if (length > 0)
    {
        file.read(
            &value[0],
            static_cast<std::streamsize>(length));
    }

    if (!file)
        throw std::runtime_error(
            "Failed to read string");

    return value;
}

Tensor readTensor(std::ifstream &file)
{
    Tensor tensor;

    tensor.name = readString(file);

    const std::uint32_t dimension_count =
        readUInt32(file);

    tensor.shape.resize(dimension_count);

    for (std::uint32_t i = 0;
         i < dimension_count;
         ++i)
    {
        tensor.shape[i] =
            readUInt64(file);
    }

    const std::uint64_t value_count =
        readUInt64(file);

    tensor.values.resize(
        static_cast<std::size_t>(value_count));

    for (std::uint64_t i = 0;
         i < value_count;
         ++i)
    {
        tensor.values[
            static_cast<std::size_t>(i)] =
            readFloat(file);
    }

    return tensor;
}

Tensor findTensor(
    const std::vector<Tensor> &tensors,
    const std::string &name)
{
    for (const Tensor &tensor : tensors)
    {
        if (tensor.name == name)
            return tensor;
    }

    throw std::runtime_error(
        "Could not find tensor: " + name);
}

std::vector<std::uint8_t> readMNISTLabels(
    const std::string &path)
{
    std::ifstream file(
        path,
        std::ios::binary);

    if (!file)
    {
        throw std::runtime_error(
            "Could not open MNIST labels: " + path);
    }

    const std::uint32_t magic =
        readUInt32BE(file);

    if (magic != 2049)
    {
        throw std::runtime_error(
            "Invalid MNIST label file");
    }

    const std::uint32_t count =
        readUInt32BE(file);

    std::vector<std::uint8_t> labels(count);

    file.read(
        reinterpret_cast<char *>(labels.data()),
        static_cast<std::streamsize>(count));

    if (!file)
        throw std::runtime_error(
            "Failed to read MNIST labels");

    return labels;
}

std::vector<std::vector<float>> readMNISTImages(
    const std::string &path)
{
    std::ifstream file(
        path,
        std::ios::binary);

    if (!file)
    {
        throw std::runtime_error(
            "Could not open MNIST images: " + path);
    }

    const std::uint32_t magic =
        readUInt32BE(file);

    if (magic != 2051)
    {
        throw std::runtime_error(
            "Invalid MNIST image file");
    }

    const std::uint32_t count =
        readUInt32BE(file);

    const std::uint32_t rows =
        readUInt32BE(file);

    const std::uint32_t columns =
        readUInt32BE(file);

    if (rows != 28 || columns != 28)
    {
        throw std::runtime_error(
            "Expected 28x28 MNIST images");
    }

    std::vector<std::vector<float>> images(
        count,
        std::vector<float>(784));

    for (std::uint32_t image = 0;
         image < count;
         ++image)
    {
        for (std::size_t pixel = 0;
             pixel < 784;
             ++pixel)
        {
            std::uint8_t value;

            file.read(
                reinterpret_cast<char *>(&value),
                sizeof(value));

            if (!file)
                throw std::runtime_error(
                    "Failed to read MNIST image");

            /*
             * Same preprocessing as train.py:
             *
             * transforms.ToTensor()
             * transforms.Normalize(
             *     (0.1307,),
             *     (0.3081,)
             * )
             *
             * ToTensor converts [0,255] -> [0,1].
             */
            const float normalized =
                (static_cast<float>(value) / 255.0f
                 - 0.1307f)
                / 0.3081f;

            images[image][pixel] =
                normalized;
        }
    }

    return images;
}

MNISTDataset loadMNIST(
    const std::string &image_path,
    const std::string &label_path)
{
    MNISTDataset dataset;

    dataset.images =
        readMNISTImages(image_path);

    dataset.labels =
        readMNISTLabels(label_path);

    if (dataset.images.size() !=
        dataset.labels.size())
    {
        throw std::runtime_error(
            "MNIST image/label count mismatch");
    }

    return dataset;
}

int argmax(
    const std::vector<float> &values)
{
    return static_cast<int>(
        std::distance(
            values.begin(),
            std::max_element(
                values.begin(),
                values.end())));
}

std::vector<float> relu(
    const std::vector<float> &values)
{
    std::vector<float> result =
        values;

    for (float &value : result)
    {
        if (value < 0.0f)
            value = 0.0f;
    }

    return result;
}

std::vector<float> fp32Linear(
    const std::vector<float> &input,
    const Tensor &weights,
    const Tensor &bias)
{
    const std::size_t output_size =
        static_cast<std::size_t>(
            weights.shape[0]);

    const std::size_t input_size =
        static_cast<std::size_t>(
            weights.shape[1]);

    if (input.size() != input_size)
    {
        throw std::runtime_error(
            "FP32 linear input size mismatch");
    }

    std::vector<float> output(
        output_size);

    for (std::size_t output_index = 0;
         output_index < output_size;
         ++output_index)
    {
        float sum =
            bias.values[output_index];

        for (std::size_t input_index = 0;
             input_index < input_size;
             ++input_index)
        {
            sum +=
                input[input_index] *
                weights.values[
                    output_index * input_size
                    + input_index];
        }

        output[output_index] =
            sum;
    }

    return output;
}

std::vector<float> fp32Inference(
    const std::vector<float> &input,
    const Tensor &fc1_weight,
    const Tensor &fc1_bias,
    const Tensor &fc2_weight,
    const Tensor &fc2_bias,
    const Tensor &fc3_weight,
    const Tensor &fc3_bias)
{
    std::vector<float> x =
        fp32Linear(
            input,
            fc1_weight,
            fc1_bias);

    x = relu(x);

    x = fp32Linear(
        x,
        fc2_weight,
        fc2_bias);

    x = relu(x);

    x = fp32Linear(
        x,
        fc3_weight,
        fc3_bias);

    return x;
}

AFPEncodedTensor encodeVector(
    const std::vector<float> &values,
    AFPQuantizer &quantizer)
{
    return quantizer.encode(values);
}

std::vector<float> afpLinear(
    const std::vector<float> &input,
    const Tensor &weights,
    const Tensor &bias,
    AFPQuantizer &quantizer)
{
    const AFPEncodedTensor afpInput =
        encodeVector(
            input,
            quantizer);

    const AFPEncodedTensor afpWeights =
        encodeVector(
            weights.values,
            quantizer);

    AFPEncodedTensor output =
        AFPArithmetic::matrixVectorMultiply(
            afpWeights,
            afpInput,
            static_cast<std::size_t>(
                weights.shape[0]),
            static_cast<std::size_t>(
                weights.shape[1]));

    const std::vector<float> decoded =
        quantizer.decode(output);

    if (decoded.size() !=
        bias.values.size())
    {
        throw std::runtime_error(
            "AFP linear output size mismatch");
    }

    /*
     * Bias is added in FP32 here intentionally.
     *
     * This first benchmark isolates the error
     * from the matrix-vector multiplication.
     *
     * Once this works, we can make the bias
     * AFP-native too.
     */
    std::vector<float> result =
        decoded;

    for (std::size_t i = 0;
         i < result.size();
         ++i)
    {
        result[i] +=
            bias.values[i];
    }

    return result;
}

std::vector<float> afpInference(
    const std::vector<float> &input,
    const Tensor &fc1_weight,
    const Tensor &fc1_bias,
    const Tensor &fc2_weight,
    const Tensor &fc2_bias,
    const Tensor &fc3_weight,
    const Tensor &fc3_bias,
    AFPQuantizer &quantizer)
{
    std::vector<float> x =
        afpLinear(
            input,
            fc1_weight,
            fc1_bias,
            quantizer);

    /*
     * ReLU.
     */
    for (float &value : x)
    {
        if (value < 0.0f)
            value = 0.0f;
    }

    x =
        afpLinear(
            x,
            fc2_weight,
            fc2_bias,
            quantizer);

    /*
     * ReLU.
     */
    for (float &value : x)
    {
        if (value < 0.0f)
            value = 0.0f;
    }

    x =
        afpLinear(
            x,
            fc3_weight,
            fc3_bias,
            quantizer);

    return x;
}

struct AccuracyResult
{
    std::size_t correct = 0;
    std::size_t total = 0;

    double accuracy = 0.0;
};

AccuracyResult calculateAccuracy(
    const std::vector<int> &predictions,
    const std::vector<std::uint8_t> &labels)
{
    AccuracyResult result;

    result.total =
        labels.size();

    for (std::size_t i = 0;
         i < labels.size();
         ++i)
    {
        if (predictions[i] ==
            static_cast<int>(labels[i]))
        {
            ++result.correct;
        }
    }

    if (result.total > 0)
    {
        result.accuracy =
            100.0 *
            static_cast<double>(result.correct) /
            static_cast<double>(result.total);
    }

    return result;
}

double calculateMAE(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error(
            "MAE size mismatch");

    double total = 0.0;

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        total +=
            std::fabs(
                static_cast<double>(a[i]) -
                static_cast<double>(b[i]));
    }

    if (a.empty())
        return 0.0;

    return total /
           static_cast<double>(a.size());
}

double calculateRMSE(
    const std::vector<float> &a,
    const std::vector<float> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error(
            "RMSE size mismatch");

    double total = 0.0;

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        const double difference =
            static_cast<double>(a[i]) -
            static_cast<double>(b[i]);

        total +=
            difference * difference;
    }

    if (a.empty())
        return 0.0;

    return std::sqrt(
        total /
        static_cast<double>(a.size()));
}

int main()
{
    try
    {
        /*
         * All paths are relative to:
         *
         * root/experiments/mnist_mlp_1/
         */
        const std::string weights_path =
            "experiments/mnist_mlp_1/mnist_mlp_weights.bin";

        const std::string images_path =
            "data/MNIST/raw/t10k-images-idx3-ubyte";

        const std::string labels_path =
            "data/MNIST/raw/t10k-labels-idx1-ubyte";

        /*
         * ========================================================
         * READ WEIGHTS
         * ========================================================
         */

        std::ifstream weight_file(
            weights_path,
            std::ios::binary);

        if (!weight_file)
        {
            throw std::runtime_error(
                "Could not open " +
                weights_path);
        }

        const std::uint32_t tensor_count =
            readUInt32(weight_file);

        std::vector<Tensor> tensors;

        tensors.reserve(
            tensor_count);

        for (std::uint32_t i = 0;
             i < tensor_count;
             ++i)
        {
            tensors.push_back(
                readTensor(weight_file));
        }

        std::cout
            << "Loaded "
            << tensor_count
            << " tensors.\n\n";

        for (const Tensor &tensor : tensors)
        {
            std::cout
                << tensor.name
                << " [";

            for (std::size_t i = 0;
                 i < tensor.shape.size();
                 ++i)
            {
                std::cout
                    << tensor.shape[i];

                if (i + 1 <
                    tensor.shape.size())
                {
                    std::cout << " x ";
                }
            }

            std::cout
                << "]\n";
        }

        /*
         * Find the six tensors belonging to
         * the three Linear layers.
         */
        const Tensor fc1_weight =
            findTensor(
                tensors,
                "fc1.weight");

        const Tensor fc1_bias =
            findTensor(
                tensors,
                "fc1.bias");

        const Tensor fc2_weight =
            findTensor(
                tensors,
                "fc2.weight");

        const Tensor fc2_bias =
            findTensor(
                tensors,
                "fc2.bias");

        const Tensor fc3_weight =
            findTensor(
                tensors,
                "fc3.weight");

        const Tensor fc3_bias =
            findTensor(
                tensors,
                "fc3.bias");

        /*
         * Validate architecture.
         */
        if (fc1_weight.shape !=
            std::vector<std::uint64_t>{128, 784})
        {
            throw std::runtime_error(
                "fc1.weight has unexpected shape");
        }

        if (fc1_bias.shape !=
            std::vector<std::uint64_t>{128})
        {
            throw std::runtime_error(
                "fc1.bias has unexpected shape");
        }

        if (fc2_weight.shape !=
            std::vector<std::uint64_t>{64, 128})
        {
            throw std::runtime_error(
                "fc2.weight has unexpected shape");
        }

        if (fc2_bias.shape !=
            std::vector<std::uint64_t>{64})
        {
            throw std::runtime_error(
                "fc2.bias has unexpected shape");
        }

        if (fc3_weight.shape !=
            std::vector<std::uint64_t>{10, 64})
        {
            throw std::runtime_error(
                "fc3.weight has unexpected shape");
        }

        if (fc3_bias.shape !=
            std::vector<std::uint64_t>{10})
        {
            throw std::runtime_error(
                "fc3.bias has unexpected shape");
        }

        /*
         * ========================================================
         * LOAD MNIST
         * ========================================================
         */

        std::cout
            << "\nLoading MNIST test set...\n";

        MNISTDataset dataset =
            loadMNIST(
                images_path,
                labels_path);

        std::cout
            << "Loaded "
            << dataset.images.size()
            << " test images.\n\n";

        /*
         * ========================================================
         * AFP SETUP
         * ========================================================
         */

        AFPConfig config;
        AFPQuantizer quantizer(config);

        /*
         * ========================================================
         * RUN FP32
         * ========================================================
         */

        std::vector<int> fp32_predictions;

        fp32_predictions.reserve(
            dataset.images.size());

        auto fp32_start =
            std::chrono::high_resolution_clock::now();

        for (const auto &image :
             dataset.images)
        {
            const std::vector<float> output =
                fp32Inference(
                    image,
                    fc1_weight,
                    fc1_bias,
                    fc2_weight,
                    fc2_bias,
                    fc3_weight,
                    fc3_bias);

            fp32_predictions.push_back(
                argmax(output));
        }

        auto fp32_end =
            std::chrono::high_resolution_clock::now();

        /*
         * ========================================================
         * RUN AFP
         * ========================================================
         */

        std::vector<int> afp_predictions;

        afp_predictions.reserve(
            dataset.images.size());

        double total_mae = 0.0;
        double total_rmse = 0.0;

        std::size_t different_predictions = 0;

        auto afp_start =
            std::chrono::high_resolution_clock::now();

        for (std::size_t image_index = 0;
             image_index < dataset.images.size();
             ++image_index)
        {
            const std::vector<float> fp32_output =
                fp32Inference(
                    dataset.images[image_index],
                    fc1_weight,
                    fc1_bias,
                    fc2_weight,
                    fc2_bias,
                    fc3_weight,
                    fc3_bias);

            const std::vector<float> afp_output =
                afpInference(
                    dataset.images[image_index],
                    fc1_weight,
                    fc1_bias,
                    fc2_weight,
                    fc2_bias,
                    fc3_weight,
                    fc3_bias,
                    quantizer);

            const int fp32_prediction =
                argmax(fp32_output);

            const int afp_prediction =
                argmax(afp_output);

            afp_predictions.push_back(
                afp_prediction);

            if (fp32_prediction !=
                afp_prediction)
            {
                ++different_predictions;
            }

            total_mae +=
                calculateMAE(
                    fp32_output,
                    afp_output);

            total_rmse +=
                calculateRMSE(
                    fp32_output,
                    afp_output);

            /*
             * Show the first few examples.
             */
            if (image_index) //< 10)
            {
                std::cout
                    << "Image "
                    << std::setw(4)
                    << image_index
                    << " | label="
                    << static_cast<int>(
                           dataset.labels[image_index])
                    << " | FP32="
                    << fp32_prediction
                    << " | AFP="
                    << afp_prediction;

                if (fp32_prediction !=
                    afp_prediction)
                {
                    std::cout
                        << "  <-- DIFFERENT";
                }

                std::cout
                    << '\n';
            }
        }

        auto afp_end =
            std::chrono::high_resolution_clock::now();

        /*
         * ========================================================
         * ACCURACY
         * ========================================================
         */

        const AccuracyResult fp32_accuracy =
            calculateAccuracy(
                fp32_predictions,
                dataset.labels);

        const AccuracyResult afp_accuracy =
            calculateAccuracy(
                afp_predictions,
                dataset.labels);

        const double fp32_time =
            std::chrono::duration<double>(
                fp32_end - fp32_start)
                .count();

        const double afp_time =
            std::chrono::duration<double>(
                afp_end - afp_start)
                .count();

        const double image_count =
            static_cast<double>(
                dataset.images.size());

        std::cout
            << "\n========================================\n";
        std::cout
            << "MNIST AFP BENCHMARK\n";
        std::cout
            << "========================================\n\n";

        std::cout
            << std::fixed
            << std::setprecision(4);

        std::cout
            << "FP32 accuracy: "
            << fp32_accuracy.accuracy
            << "% ("
            << fp32_accuracy.correct
            << "/"
            << fp32_accuracy.total
            << ")\n";

        std::cout
            << "AFP accuracy:  "
            << afp_accuracy.accuracy
            << "% ("
            << afp_accuracy.correct
            << "/"
            << afp_accuracy.total
            << ")\n";

        std::cout
            << "Accuracy loss: "
            << fp32_accuracy.accuracy -
                   afp_accuracy.accuracy
            << " percentage points\n";

        std::cout
            << "\n";

        std::cout
            << "AFP vs FP32 different predictions: "
            << different_predictions
            << " / "
            << dataset.images.size()
            << " ("
            << 100.0 *
                   static_cast<double>(
                       different_predictions) /
                   image_count
            << "%)\n";

        std::cout
            << "\n";

        std::cout
            << "Average output MAE:  "
            << total_mae / image_count
            << '\n';

        std::cout
            << "Average output RMSE: "
            << total_rmse / image_count
            << '\n';

        std::cout
            << "\n";

        std::cout
            << "FP32 inference time: "
            << fp32_time
            << " seconds\n";

        std::cout
            << "AFP inference time:  "
            << afp_time
            << " seconds\n";

        std::cout
            << "FP32 images/sec:     "
            << image_count / fp32_time
            << '\n';

        std::cout
            << "AFP images/sec:      "
            << image_count / afp_time
            << '\n';

        std::cout
            << "\n========================================\n";
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "\nERROR: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}