#include "ec_pro_native/inference_engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "ggml-backend.h"

namespace ec_pro_native {

namespace {

std::once_flag backend_once;

void initialise_backend() {
    std::call_once(
        backend_once,
        []() {
            ggml_backend_load_all();
        }
    );
}

}  // namespace

InferenceEngine::InferenceEngine(
    EngineConfig config
)
    : config_(std::move(config)) {
    if (config_.model_path.empty()) {
        throw std::invalid_argument(
            "model_path cannot be empty"
        );
    }

    if (config_.n_ctx < 128) {
        throw std::invalid_argument(
            "n_ctx must be at least 128"
        );
    }

    initialise_backend();

    auto model_params =
        llama_model_default_params();

    model_params.n_gpu_layers =
        config_.n_gpu_layers;

    model_ = llama_model_load_from_file(
        config_.model_path.c_str(),
        model_params
    );

    if (model_ == nullptr) {
        throw std::runtime_error(
            "Failed to load GGUF model: "
            + config_.model_path
        );
    }

    vocab_ = llama_model_get_vocab(
        model_
    );

    if (vocab_ == nullptr) {
        llama_model_free(model_);
        model_ = nullptr;

        throw std::runtime_error(
            "Model does not expose a vocabulary"
        );
    }

    token_processor_ =
        std::make_unique<TokenProcessor>(
            vocab_
        );
}

InferenceEngine::~InferenceEngine() {
    token_processor_.reset();

    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

const EngineConfig&
InferenceEngine::config() const noexcept {
    return config_;
}

std::string
InferenceEngine::render_chat(
    const std::vector<ChatMessage>& messages
) const {
    if (messages.empty()) {
        throw std::invalid_argument(
            "At least one message is required"
        );
    }

    std::vector<llama_chat_message>
        native_messages;

    native_messages.reserve(
        messages.size()
    );

    for (const auto& message : messages) {
        native_messages.push_back(
            {
                message.role.c_str(),
                message.content.c_str()
            }
        );
    }

    const char* chat_template =
        llama_model_chat_template(
            model_,
            nullptr
        );

    if (chat_template == nullptr) {
        throw std::runtime_error(
            "GGUF model has no chat template"
        );
    }

    auto required =
        llama_chat_apply_template(
            chat_template,
            native_messages.data(),
            native_messages.size(),
            true,
            nullptr,
            0
        );

    if (required < 0) {
        throw std::runtime_error(
            "Failed to calculate chat template size"
        );
    }

    std::vector<char> buffer(
        static_cast<std::size_t>(
            required
        ) + 1
    );

    const auto written =
        llama_chat_apply_template(
            chat_template,
            native_messages.data(),
            native_messages.size(),
            true,
            buffer.data(),
            static_cast<std::int32_t>(
                buffer.size()
            )
        );

    if (written < 0) {
        throw std::runtime_error(
            "Failed to render chat template"
        );
    }

    return std::string(
        buffer.data(),
        static_cast<std::size_t>(
            written
        )
    );
}

GenerationResult
InferenceEngine::generate(
    const GenerationRequest& request
) const {
    const auto started =
        std::chrono::steady_clock::now();

    const auto prompt =
        render_chat(
            request.messages
        );

    const auto prompt_tokens =
        token_processor_->tokenize(
            prompt,
            true,
            true
        );

    if (prompt_tokens.empty()) {
        throw std::runtime_error(
            "Prompt produced no tokens"
        );
    }

    const auto required_context =
        prompt_tokens.size()
        + static_cast<std::size_t>(
            request.generation
                .max_new_tokens
        );

    if (
        required_context
        > config_.n_ctx
    ) {
        throw std::runtime_error(
            "Prompt + generated tokens "
            "exceed configured context size"
        );
    }

    auto context_params =
        llama_context_default_params();

    context_params.n_ctx =
        config_.n_ctx;

    context_params.n_batch =
        config_.n_batch;

    if (config_.n_threads > 0) {
        context_params.n_threads =
            config_.n_threads;
    }

    if (
        config_.n_threads_batch > 0
    ) {
        context_params.n_threads_batch =
            config_.n_threads_batch;
    }

    llama_context* context =
        llama_init_from_model(
            model_,
            context_params
        );

    if (context == nullptr) {
        throw std::runtime_error(
            "Failed to create llama context"
        );
    }

    llama_sampler* sampler = nullptr;

    try {
        if (
            request.generation.temperature
            <= 0.0F
        ) {
            sampler =
                llama_sampler_init_greedy();
        }
        else {
            sampler =
                llama_sampler_chain_init(
                    llama_sampler_chain_default_params()
                );

            llama_sampler_chain_add(
                sampler,
                llama_sampler_init_top_p(
                    request.generation.top_p,
                    1
                )
            );

            llama_sampler_chain_add(
                sampler,
                llama_sampler_init_temp(
                    request.generation
                        .temperature
                )
            );

            const auto seed =
                request.generation.seed
                    .value_or(
                        LLAMA_DEFAULT_SEED
                    );

            llama_sampler_chain_add(
                sampler,
                llama_sampler_init_dist(
                    seed
                )
            );
        }

        if (sampler == nullptr) {
            throw std::runtime_error(
                "Failed to create sampler"
            );
        }

        auto prompt_copy =
            prompt_tokens;

        auto batch =
            llama_batch_get_one(
                prompt_copy.data(),
                static_cast<std::int32_t>(
                    prompt_copy.size()
                )
            );

        if (
            llama_decode(
                context,
                batch
            ) != 0
        ) {
            throw std::runtime_error(
                "Failed to decode prompt"
            );
        }

        std::string output;
        std::size_t generated_count = 0;

        for (
            std::int32_t step = 0;
            step
                < request.generation
                      .max_new_tokens;
            ++step
        ) {
            const auto token =
                llama_sampler_sample(
                    sampler,
                    context,
                    -1
                );

            if (
                llama_vocab_is_eog(
                    vocab_,
                    token
                )
            ) {
                break;
            }

            output +=
                token_processor_
                    ->token_to_piece(
                        token
                    );

            ++generated_count;

            auto next_token = token;

            auto next_batch =
                llama_batch_get_one(
                    &next_token,
                    1
                );

            if (
                llama_decode(
                    context,
                    next_batch
                ) != 0
            ) {
                throw std::runtime_error(
                    "Failed to decode generated token"
                );
            }
        }

        const auto finished =
            std::chrono::steady_clock::now();

        GenerationResult result;

        result.text =
            std::move(output);

        result.prompt_tokens =
            prompt_tokens.size();

        result.generated_tokens =
            generated_count;

        result.latency_ms =
            std::chrono::duration_cast<
                std::chrono::milliseconds
            >(
                finished - started
            ).count();

        llama_sampler_free(
            sampler
        );

        llama_free(
            context
        );

        return result;
    }
    catch (...) {
        if (sampler != nullptr) {
            llama_sampler_free(
                sampler
            );
        }

        llama_free(
            context
        );

        throw;
    }
}

std::vector<GenerationResult>
InferenceEngine::generate_batch(
    const std::vector<
        GenerationRequest
    >& requests,
    std::size_t parallelism
) const {
    std::vector<GenerationResult>
        results(requests.size());

    if (requests.empty()) {
        return results;
    }

    if (parallelism == 0) {
        parallelism = 1;
    }

    parallelism = std::min(
        parallelism,
        requests.size()
    );

    if (parallelism == 1) {
        for (
            std::size_t index = 0;
            index < requests.size();
            ++index
        ) {
            results[index] =
                generate(
                    requests[index]
                );
        }

        return results;
    }

    std::atomic<std::size_t>
        next_index{0};

    std::mutex error_mutex;
    std::exception_ptr error;

    std::vector<std::thread> workers;

    workers.reserve(
        parallelism
    );

    for (
        std::size_t worker = 0;
        worker < parallelism;
        ++worker
    ) {
        workers.emplace_back(
            [&, this]() {
                while (true) {
                    const auto index =
                        next_index.fetch_add(1);

                    if (
                        index
                        >= requests.size()
                    ) {
                        return;
                    }

                    try {
                        results[index] =
                            generate(
                                requests[index]
                            );
                    }
                    catch (...) {
                        std::lock_guard lock(
                            error_mutex
                        );

                        if (!error) {
                            error =
                                std::current_exception();
                        }

                        return;
                    }
                }
            }
        );
    }

    for (auto& worker : workers) {
        worker.join();
    }

    if (error) {
        std::rethrow_exception(
            error
        );
    }

    return results;
}

std::vector<llama_token>
InferenceEngine::tokenize(
    const std::string& text
) const {
    return token_processor_->tokenize(
        text
    );
}

std::vector<
    std::vector<llama_token>
>
InferenceEngine::tokenize_many(
    const std::vector<std::string>& texts,
    std::size_t worker_count
) const {
    return token_processor_
        ->tokenize_many(
            texts,
            worker_count
        );
}

}  // namespace ec_pro_native