#include <iostream>
#include <stdexcept>
#include <string>

#include "kirk_gpt_native/token_processor.hpp"

namespace {

using namespace kirk_gpt_native;

void require(
    bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(
            message
        );
    }
}


void test_null_vocab_rejected() {
    bool rejected = false;

    try {
        TokenProcessor processor(
            nullptr
        );

        (void) processor;
    }
    catch (
        const std::invalid_argument&
            error
    ) {
        rejected =
            std::string(
                error.what()
            ).find(
                "valid vocabulary"
            ) != std::string::npos;
    }

    require(
        rejected,
        "TokenProcessor accepted "
        "a null llama vocabulary"
    );
}

}  // namespace


int main() {
    try {
        test_null_vocab_rejected();

        std::cout
            << "[PASS] null vocabulary "
               "rejected\n";

        std::cout
            << "1/1 model-free token "
               "processor tests passed\n";

        return 0;
    }
    catch (
        const std::exception& error
    ) {
        std::cerr
            << "[FAIL] token processor: "
            << error.what()
            << '\n';

        return 1;
    }
}