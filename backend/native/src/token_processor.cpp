#include "ec_pro_native/token_processor.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ec_pro_native {

namespace {

std::int32_t checked_int32_size(
    std::size_t size,
    const char* name
) {
    constexpr auto max_int32 =
        static_cast<std::size_t>(
            std::numeric_limits<
                std::int32_t
            >::max()
        );

    if (size > max_int32) {
        throw std::length_error(
            std::string(name)
            + " exceeds llama.cpp "
              "int32 size limit"
        );
    }

    return static_cast<std::int32_t>(
        size
    );
}

std::size_t required_capacity(
    std::int32_t result,
    const char* operation
) {
    if (result >= 0) {
        return static_cast<std::size_t>(
            result
        );
    }

    // Convert before negating so INT32_MIN
    // cannot overflow.
    const auto required =
        -static_cast<std::int64_t>(
            result
        );

    if (
        required <= 0
        || required
            > std::numeric_limits<
                  std::int32_t
              >::max()
    ) {
        throw std::runtime_error(
            std::string(operation)
            + " returned an invalid "
              "required size"
        );
    }

    return static_cast<std::size_t>(
        required
    );
}

}  // namespace


TokenProcessor::TokenProcessor(
    const llama_vocab* vocab
)
    : vocab_(vocab) {
    if (vocab_ == nullptr) {
        throw std::invalid_argument(
            "TokenProcessor requires "
            "a valid vocabulary"
        );
    }
}


std::vector<llama_token>
TokenProcessor::tokenize(
    const std::string& text,
    bool add_special,
    bool parse_special
) const {
    const auto text_length =
        checked_int32_size(
            text.size(),
            "Tokenization input"
        );

    const auto required =
        llama_tokenize(
            vocab_,
            text.data(),
            text_length,
            nullptr,
            0,
            add_special,
            parse_special
        );

    if (required == 0) {
        return {};
    }

    const auto token_capacity =
        required_capacity(
            required,
            "llama_tokenize"
        );

    if (token_capacity == 0) {
        return {};
    }

    std::vector<llama_token> tokens(
        token_capacity
    );

    const auto written =
        llama_tokenize(
            vocab_,
            text.data(),
            text_length,
            tokens.data(),
            checked_int32_size(
                tokens.size(),
                "Token buffer"
            ),
            add_special,
            parse_special
        );

    if (written < 0) {
        throw std::runtime_error(
            "llama_tokenize failed "
            "after allocating the "
            "requested token buffer"
        );
    }

    const auto written_size =
        static_cast<std::size_t>(
            written
        );

    if (
        written_size
        > tokens.size()
    ) {
        throw std::runtime_error(
            "llama_tokenize wrote more "
            "tokens than the allocated "
            "buffer"
        );
    }

    tokens.resize(
        written_size
    );

    return tokens;
}


std::string
TokenProcessor::token_to_piece(
    llama_token token
) const {
    // Most token pieces are tiny, so avoid
    // a heap allocation in the common case.
    std::array<char, 128>
        local_buffer{};

    auto written =
        llama_token_to_piece(
            vocab_,
            token,
            local_buffer.data(),
            static_cast<std::int32_t>(
                local_buffer.size()
            ),
            0,
            true
        );

    if (written >= 0) {
        const auto written_size =
            static_cast<std::size_t>(
                written
            );

        if (
            written_size
            > local_buffer.size()
        ) {
            throw std::runtime_error(
                "llama_token_to_piece "
                "reported an invalid "
                "written size"
            );
        }

        return std::string(
            local_buffer.data(),
            written_size
        );
    }

    // llama.cpp reports the required
    // capacity as a negative value when
    // the supplied buffer is too small.
    const auto capacity =
        required_capacity(
            written,
            "llama_token_to_piece"
        );

    std::vector<char> buffer(
        capacity
    );

    written =
        llama_token_to_piece(
            vocab_,
            token,
            buffer.data(),
            checked_int32_size(
                buffer.size(),
                "Token piece buffer"
            ),
            0,
            true
        );

    if (written < 0) {
        throw std::runtime_error(
            "llama_token_to_piece failed "
            "after resizing its buffer"
        );
    }

    const auto written_size =
        static_cast<std::size_t>(
            written
        );

    if (
        written_size
        > buffer.size()
    ) {
        throw std::runtime_error(
            "llama_token_to_piece wrote "
            "more bytes than the "
            "allocated buffer"
        );
    }

    return std::string(
        buffer.data(),
        written_size
    );
}


std::string
TokenProcessor::detokenize(
    const std::vector<
        llama_token
    >& tokens
) const {
    std::string output;

    for (const auto token : tokens) {
        output +=
            token_to_piece(
                token
            );
    }

    return output;
}


std::vector<
    std::vector<llama_token>
>
TokenProcessor::tokenize_many(
    const std::vector<std::string>& texts,
    std::size_t worker_count
) const {
    std::vector<
        std::vector<llama_token>
    > results(
        texts.size()
    );

    if (texts.empty()) {
        return results;
    }

    if (worker_count == 0) {
        worker_count =
            std::thread::
                hardware_concurrency();

        if (worker_count == 0) {
            worker_count = 1;
        }
    }

    worker_count =
        std::min(
            worker_count,
            texts.size()
        );

    // Avoid thread creation overhead for
    // explicitly serial or one-item work.
    if (worker_count == 1) {
        for (
            std::size_t index = 0;
            index < texts.size();
            ++index
        ) {
            results[index] =
                tokenize(
                    texts[index]
                );
        }

        return results;
    }

    std::atomic<std::size_t>
        next_index{0};

    std::atomic<bool>
        stop{false};

    std::mutex error_mutex;
    std::exception_ptr error;

    std::vector<std::thread>
        workers;

    workers.reserve(
        worker_count
    );

    for (
        std::size_t worker = 0;
        worker < worker_count;
        ++worker
    ) {
        workers.emplace_back(
            [&, this]() {
                while (
                    !stop.load(
                        std::memory_order_relaxed
                    )
                ) {
                    const auto index =
                        next_index.fetch_add(
                            1,
                            std::memory_order_relaxed
                        );

                    if (
                        index
                        >= texts.size()
                    ) {
                        return;
                    }

                    try {
                        results[index] =
                            tokenize(
                                texts[index]
                            );
                    }
                    catch (...) {
                        {
                            std::lock_guard lock(
                                error_mutex
                            );

                            if (!error) {
                                error =
                                    std::
                                        current_exception();
                            }
                        }

                        stop.store(
                            true,
                            std::memory_order_relaxed
                        );

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

}  // namespace ec_pro_native