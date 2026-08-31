#include "ec_pro_native/token_processor.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace ec_pro_native {

TokenProcessor::TokenProcessor(
    const llama_vocab* vocab
)
    : vocab_(vocab) {
    if (vocab_ == nullptr) {
        throw std::invalid_argument(
            "TokenProcessor requires a valid vocabulary"
        );
    }
}

std::vector<llama_token>
TokenProcessor::tokenize(
    const std::string& text,
    bool add_special,
    bool parse_special
) const {
    const auto required = llama_tokenize(
        vocab_,
        text.data(),
        static_cast<std::int32_t>(
            text.size()
        ),
        nullptr,
        0,
        add_special,
        parse_special
    );

    if (required == 0) {
        return {};
    }

    const auto token_capacity =
        required < 0
            ? -required
            : required;

    std::vector<llama_token> tokens(
        static_cast<std::size_t>(
            token_capacity
        )
    );

    const auto written = llama_tokenize(
        vocab_,
        text.data(),
        static_cast<std::int32_t>(
            text.size()
        ),
        tokens.data(),
        static_cast<std::int32_t>(
            tokens.size()
        ),
        add_special,
        parse_special
    );

    if (written < 0) {
        throw std::runtime_error(
            "llama_tokenize failed"
        );
    }

    tokens.resize(
        static_cast<std::size_t>(
            written
        )
    );

    return tokens;
}

std::string
TokenProcessor::token_to_piece(
    llama_token token
) const {
    std::vector<char> buffer(1024);

    const auto written =
        llama_token_to_piece(
            vocab_,
            token,
            buffer.data(),
            static_cast<std::int32_t>(
                buffer.size()
            ),
            0,
            true
        );

    if (written < 0) {
        throw std::runtime_error(
            "llama_token_to_piece failed"
        );
    }

    return std::string(
        buffer.data(),
        static_cast<std::size_t>(
            written
        )
    );
}

std::string
TokenProcessor::detokenize(
    const std::vector<llama_token>& tokens
) const {
    std::string output;

    for (const auto token : tokens) {
        output += token_to_piece(token);
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
    > results(texts.size());

    if (texts.empty()) {
        return results;
    }

    if (worker_count == 0) {
        worker_count =
            std::thread::hardware_concurrency();

        if (worker_count == 0) {
            worker_count = 1;
        }
    }

    worker_count = std::min(
        worker_count,
        texts.size()
    );

    std::atomic<std::size_t> next_index{0};

    std::mutex error_mutex;
    std::exception_ptr error;

    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (
        std::size_t worker = 0;
        worker < worker_count;
        ++worker
    ) {
        workers.emplace_back(
            [&, this]() {
                while (true) {
                    const auto index =
                        next_index.fetch_add(1);

                    if (
                        index >= texts.size()
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
        std::rethrow_exception(error);
    }

    return results;
}

}  // namespace ec_pro_native