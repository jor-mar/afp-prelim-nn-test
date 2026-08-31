#pragma once

#include <vector>
#include <string>

class Quantizer
{
public:
    virtual ~Quantizer() = default;

    virtual std::vector<float> quantize(
        const std::vector<float>& input
    ) const = 0;

    virtual std::string name() const = 0;
    virtual double bitsPerValue() const = 0;
};