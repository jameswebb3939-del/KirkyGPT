#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "llama.h"

#include "ec_pro_native/generation_backend.hpp"
#include "ec_pro_native/token_processor.hpp"
#include "ec_pro_native/types.hpp"

namespace ec_pro_native {

class InferenceEngine final
    : public GenerationBackend {
public:
    explicit InferenceEngine(
        EngineConfig config
    );

    ~InferenceEngine();

    InferenceEngine(
        const InferenceEngine&
    ) = delete;

    InferenceEngine& operator=(
        const InferenceEngine&
    ) = delete;

    std::string render_chat(
        const std::vector<ChatMessage>& messages
    ) const;

    GenerationResult generate(
        const GenerationRequest& request
    ) const;

    std::vector<GenerationResult>
    generate_batch(
        const std::vector<
            GenerationRequest
        >& requests,
        std::size_t parallelism = 1
    ) const override;

    std::vector<llama_token> tokenize(
        const std::string& text
    ) const;

    std::vector<
        std::vector<llama_token>
    > tokenize_many(
        const std::vector<std::string>& texts,
        std::size_t worker_count = 0
    ) const;

    const EngineConfig&
    config() const noexcept;

private:
    EngineConfig config_;

    llama_model* model_ = nullptr;
    const llama_vocab* vocab_ = nullptr;

    std::unique_ptr<TokenProcessor>
        token_processor_;
};

}  // namespace ec_pro_native