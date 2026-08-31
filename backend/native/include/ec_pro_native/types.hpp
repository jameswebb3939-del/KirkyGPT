#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ec_pro_native {

struct EngineConfig {
    std::string model_path;

    std::uint32_t n_ctx = 2048;
    std::uint32_t n_batch = 512;

    std::int32_t n_threads = 0;
    std::int32_t n_threads_batch = 0;

    std::int32_t n_gpu_layers = 0;
};

struct GenerationConfig {
    std::int32_t max_new_tokens = 64;

    float temperature = 0.2F;
    float top_p = 0.9F;

    std::optional<std::uint32_t> seed;
};

struct ChatMessage {
    std::string role;
    std::string content;
};

struct GenerationRequest {
    std::vector<ChatMessage> messages;
    GenerationConfig generation;
};

struct GenerationResult {
    std::string text;

    std::int64_t latency_ms = 0;

    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
};

}  // namespace ec_pro_native