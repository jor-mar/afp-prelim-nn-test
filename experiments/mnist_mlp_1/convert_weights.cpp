#include "../../include/afp_encoded_tensor.hpp"
#include "../../include/afp_math.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

struct TensorData
{
    std::string name;
    std::vector<std::uint64_t>(shape);
    std::vector<float> values;
};

static std::uint32_t readUint32(std::ifstream &file)
{
    std::uint32_t value = 0;

    file.read(
        reinterpret_cast<char *>(&value),
        sizeof(value));

    return value;
}

static std::uint64_t readUint64(std::ifstream &file)
{
    std::uint64_t value = 0;

    file.read(
        reinterpret_cast<char *>(&value),
        sizeof(value));

    return value;
}

static float readFloat(std::ifstream &file)
{
    float value = 0.0f;

    file.read(
        reinterpret_cast<char *>(&value),
        sizeof(value));

    return value;
}

static void writeUint32(
    std::ofstream &file,
    std::uint32_t value)
{
    file.write(
        reinterpret_cast<const char *>(&value),
        sizeof(value));
}

static void writeUint64(
    std::ofstream &file,
    std::uint64_t value)
{
    file.write(
        reinterpret_cast<const char *>(&value),
        sizeof(value));
}

static void writeUint8(
    std::ofstream &file,
    std::uint8_t value)
{
    file.write(
        reinterpret_cast<const char *>(&value),
        sizeof(value));
}

static bool readFP32Model(
    const std::string &filename,
    std::vector<TensorData> &tensors)
{
    std::ifstream input(
        filename,
        std::ios::binary);

    if (!input)
    {
        std::cerr
            << "Failed to open FP32 model: "
            << filename
            << '\n';

        return false;
    }

    std::uint32_t tensor_count = readUint32(input);

    if (!input)
    {
        std::cerr
            << "Failed to read tensor count.\n";

        return false;
    }

    tensors.clear();
    tensors.reserve(tensor_count);

    for (std::uint32_t i = 0; i < tensor_count; ++i)
    {
        TensorData tensor;

        std::uint32_t name_length =
            readUint32(input);

        tensor.name.resize(name_length);

        input.read(
            tensor.name.data(),
            static_cast<std::streamsize>(name_length));

        if (!input)
        {
            std::cerr
                << "Failed to read tensor name.\n";

            return false;
        }

        std::uint32_t dimension_count =
            readUint32(input);

        tensor.shape.resize(dimension_count);

        for (std::uint32_t d = 0; d < dimension_count; ++d)
        {
            tensor.shape[d] =
                readUint64(input);
        }

        std::uint64_t value_count =
            readUint64(input);

        tensor.values.resize(
            static_cast<std::size_t>(value_count));

        for (std::uint64_t j = 0; j < value_count; ++j)
        {
            tensor.values[static_cast<std::size_t>(j)] =
                readFloat(input);
        }

        if (!input)
        {
            std::cerr
                << "Failed to read tensor data for "
                << tensor.name
                << ".\n";

            return false;
        }

        tensors.push_back(
            std::move(tensor));
    }

    return true;
}

static bool writeAFPModel(
    const std::string &filename,
    const std::vector<TensorData> &tensors,
    const AFPConfig &config)
{
    std::ofstream output(
        filename,
        std::ios::binary);

    if (!output)
    {
        std::cerr
            << "Failed to create AFP model: "
            << filename
            << '\n';

        return false;
    }

    /*
        File format:

        4 bytes:
            magic = "AFP8"

        uint32:
            version

        uint32:
            tensor count

        AFPConfig:
            uint64 block_size
            uint32 exponent_bits
            uint32 characterization_bits
            uint32 offset_bits
            uint32 mantissa_bits
            uint8  enable_positive_fields
            uint8  enable_zero_fields

        For each tensor:
            uint32 name length
            name bytes

            uint32 dimension count
            uint64 shape[d]

            uint64 value count
            uint64 bit count
            uint64 byte count
            uint64 block count

            uint64 block_offsets[block count]

            raw bitstream bytes
    */

    const char magic[4] = {'A', 'F', 'P', '8'};

    output.write(
        magic,
        sizeof(magic));

    writeUint32(
        output,
        2);

    writeUint32(
        output,
        static_cast<std::uint32_t>(
            tensors.size()));

    writeUint64(
        output,
        static_cast<std::uint64_t>(
            config.block_size));

    writeUint32(
        output,
        static_cast<std::uint32_t>(
            config.exponent_bits));

    writeUint32(
        output,
        static_cast<std::uint32_t>(
            config.characterization_bits));

    writeUint32(
        output,
        static_cast<std::uint32_t>(
            config.offset_bits));

    writeUint32(
        output,
        static_cast<std::uint32_t>(
            config.mantissa_bits));

    writeUint8(
        output,
        config.enable_positive_fields
            ? 1
            : 0);

    writeUint8(
        output,
        config.enable_zero_fields
            ? 1
            : 0);

    AFPQuantizer quantizer(config);

    /*
        Compression totals.

        These measure the numerical tensor data only:
            FP32 = 4 bytes per value
            AFP  = encoded bitstream bytes
    */

    std::uint64_t total_values = 0;
    std::uint64_t total_fp32_bytes = 0;
    std::uint64_t total_afp_bytes = 0;

    for (const TensorData &tensor : tensors)
    {
        std::cout
            << "Encoding "
            << tensor.name
            << "...\n";

        AFPEncodedTensor encoded =
            quantizer.encode(tensor.values);

        /*
            Tensor metadata.
        */

        writeUint32(
            output,
            static_cast<std::uint32_t>(
                tensor.name.size()));

        output.write(
            tensor.name.data(),
            static_cast<std::streamsize>(
                tensor.name.size()));

        writeUint32(
            output,
            static_cast<std::uint32_t>(
                tensor.shape.size()));

        for (std::uint64_t dimension : tensor.shape)
        {
            writeUint64(
                output,
                dimension);
        }

        writeUint64(
            output,
            static_cast<std::uint64_t>(
                encoded.size()));

        writeUint64(
            output,
            static_cast<std::uint64_t>(
                encoded.bitSize()));

        writeUint64(
            output,
            static_cast<std::uint64_t>(
                encoded.byteSize()));

        writeUint64(
            output,
            static_cast<std::uint64_t>(
                encoded.blockCount()));

        /*
            Store block offsets so the AFP tensor
            can be reconstructed exactly when loaded.
        */

        for (std::size_t offset :
             encoded.blockOffsets())
        {
            writeUint64(
                output,
                static_cast<std::uint64_t>(
                    offset));
        }

        /*
            Store the actual AFP bitstream.
        */

        const std::vector<std::uint8_t> &data =
            encoded.bitStream().data();

        if (!data.empty())
        {
            output.write(
                reinterpret_cast<const char *>(
                    data.data()),
                static_cast<std::streamsize>(
                    data.size()));
        }

        if (!output)
        {
            std::cerr
                << "Failed while writing tensor "
                << tensor.name
                << ".\n";

            return false;
        }

        /*
            Calculate compression statistics.
        */

        const std::uint64_t value_count =
            static_cast<std::uint64_t>(
                tensor.values.size());

        const std::uint64_t fp32_bytes =
            value_count * sizeof(float);

        const std::uint64_t afp_bytes =
            static_cast<std::uint64_t>(
                encoded.byteSize());

        const std::int64_t bytes_saved =
            static_cast<std::int64_t>(fp32_bytes) - static_cast<std::int64_t>(afp_bytes);

        double savings_percent = 0.0;

        if (fp32_bytes > 0)
        {
            savings_percent =
                (static_cast<double>(bytes_saved) / static_cast<double>(fp32_bytes)) * 100.0;
        }

        double compression_ratio = 0.0;

        if (afp_bytes > 0)
        {
            compression_ratio =
                static_cast<double>(fp32_bytes) / static_cast<double>(afp_bytes);
        }

        total_values += value_count;
        total_fp32_bytes += fp32_bytes;
        total_afp_bytes += afp_bytes;

        std::cout
            << "  Values: "
            << encoded.size()
            << '\n';

        std::cout
            << "  Bits: "
            << encoded.bitSize()
            << '\n';

        std::cout
            << "  Bytes: "
            << encoded.byteSize()
            << '\n';

        std::cout
            << "  Bits/value: "
            << encoded.bitsPerValue()
            << '\n';

        std::cout
            << "  Blocks: "
            << encoded.blockCount()
            << '\n';

        std::cout
            << "  FP32 size: "
            << fp32_bytes
            << " bytes\n";

        std::cout
            << "  AFP size: "
            << afp_bytes
            << " bytes\n";

        std::cout
            << "  Savings: "
            << bytes_saved
            << " bytes ("
            << std::fixed
            << std::setprecision(2)
            << savings_percent
            << "%)\n";

        std::cout
            << "  Compression ratio: "
            << std::fixed
            << std::setprecision(2)
            << compression_ratio
            << "x\n";

        std::cout << '\n';
    }

    /*
        Print total compression statistics.
    */

    const std::int64_t total_bytes_saved =
        static_cast<std::int64_t>(total_fp32_bytes) - static_cast<std::int64_t>(total_afp_bytes);

    double total_savings_percent = 0.0;

    if (total_fp32_bytes > 0)
    {
        total_savings_percent =
            (static_cast<double>(total_bytes_saved) / static_cast<double>(total_fp32_bytes)) * 100.0;
    }

    double total_compression_ratio = 0.0;

    if (total_afp_bytes > 0)
    {
        total_compression_ratio =
            static_cast<double>(total_fp32_bytes) / static_cast<double>(total_afp_bytes);
    }

    std::cout
        << "========================================\n";

    std::cout
        << "TOTAL COMPRESSION\n";

    std::cout
        << "========================================\n";

    std::cout
        << "Total values: "
        << total_values
        << '\n';

    std::cout
        << "FP32 size: "
        << total_fp32_bytes
        << " bytes\n";

    std::cout
        << "AFP size: "
        << total_afp_bytes
        << " bytes\n";

    std::cout
        << "Bytes saved: "
        << total_bytes_saved
        << " bytes\n";

    std::cout
        << "Storage saved: "
        << std::fixed
        << std::setprecision(2)
        << total_savings_percent
        << "%\n";

    std::cout
        << "Compression ratio: "
        << std::fixed
        << std::setprecision(2)
        << total_compression_ratio
        << "x\n";

    std::cout
        << "FP32 bits/value: "
        << 32.0
        << '\n';

    if (total_values > 0)
    {
        const double total_afp_bits_per_value =
            (static_cast<double>(total_afp_bytes) * 8.0) / static_cast<double>(total_values);

        std::cout
            << "AFP bits/value: "
            << std::fixed
            << std::setprecision(2)
            << total_afp_bits_per_value
            << '\n';
    }

    std::cout
        << "========================================\n\n";

    return true;
}

int main()
{
    const std::string input_filename =
        "experiments/mnist_mlp_1/mnist_mlp_weights.bin";

    const std::string output_filename =
        "experiments/mnist_mlp_1/mnist_mlp_weights_afp.bin";

    /*
        Use the normal AFP8 configuration.
    */

    AFPConfig config;

    config.block_size = 16;
    config.exponent_bits = 8;
    config.characterization_bits = 8;
    config.offset_bits = 3;
    config.mantissa_bits = 5;
    config.enable_positive_fields = true;
    config.enable_zero_fields = false;

    /*
        Read the original FP32 model.
    */

    std::vector<TensorData> tensors;

    if (!readFP32Model(
            input_filename,
            tensors))
    {
        return 1;
    }

    std::cout
        << "Loaded FP32 model.\n";

    std::cout
        << "Tensor count: "
        << tensors.size()
        << "\n\n";

    /*
        Convert and save the AFP model.
    */

    if (!writeAFPModel(
            output_filename,
            tensors,
            config))
    {
        return 1;
    }

    std::cout
        << "AFP model written to: "
        << output_filename
        << '\n';

    return 0;
}