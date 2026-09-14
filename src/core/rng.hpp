#ifndef __SD_CORE_RNG_HPP__
#define __SD_CORE_RNG_HPP__

#include <random>
#include <vector>

#include "stable-diffusion.h"  // for SD_API, str_to_rng_type() in denoiser

class RNG {
public:
    virtual void manual_seed(uint64_t seed)      = 0;
    virtual std::vector<float> randn(uint32_t n) = 0;

    virtual const char* const rn() const { return "rng"; }
    virtual const std::shared_ptr<RNG> clone() const = 0;
};

extern SD_API std::shared_ptr<RNG> get_rng(rng_type_t rng_type);

class STDDefaultRNG : public RNG {
private:
    std::default_random_engine generator;

public:
    virtual const char* const rn() const override { return "std"; }
    virtual const std::shared_ptr<RNG> clone() const override {
        return std::make_shared<STDDefaultRNG>(*this);
    }

    void manual_seed(uint64_t seed) override {
        generator.seed((unsigned int)seed);
    }

    std::vector<float> randn(uint32_t n) override {
        std::vector<float> result;
        float mean   = 0.0;
        float stddev = 1.0;
        std::normal_distribution<float> distribution(mean, stddev);
        for (uint32_t i = 0; i < n; i++) {
            float random_number = distribution(generator);
            result.push_back(random_number);
        }
        return result;
    }
};

#endif  // __SD_CORE_RNG_HPP__