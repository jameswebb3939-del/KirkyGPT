#pragma once

#include <cstddef>
#include <vector>

#include "ec_pro_native/types.hpp"

namespace ec_pro_native {

class GenerationBackend {
public:
    virtual ~GenerationBackend() = default;

    virtual std::vector<GenerationResult>
    generate_batch(
        const std::vector<GenerationRequest>& requests,
        std::size_t parallelism = 1
    ) const = 0;
};

}  // namespace ec_pro_native