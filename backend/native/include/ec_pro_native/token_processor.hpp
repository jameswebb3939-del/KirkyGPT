#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "llama.h"

namespace ec_pro_native {

class TokenProcessor {
public:
    explicit TokenProcessor(
        const llama_vocab* vocab
    );

    std::vector<llama_token> tokenize(
        const std::string& text,
        bool add_special = true,
        bool parse_special = true
    ) const;

    std::string token_to_piece(
        llama_token token
    ) const;

    std::string detokenize(
        const std::vector<llama_token>& tokens
    ) const;

    std::vector<
        std::vector<llama_token>
    > tokenize_many(
        const std::vector<std::string>& texts,
        std::size_t worker_count = 0
    ) const;

private:
    const llama_vocab* vocab_;
};

}  // namespace ec_pro_native