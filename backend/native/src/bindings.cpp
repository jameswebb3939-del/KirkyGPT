#include <memory>
#include <string>
#include <stdexcept>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ec_pro_native/batch_scheduler.hpp"
#include "ec_pro_native/inference_engine.hpp"
#include "ec_pro_native/types.hpp"

namespace py = pybind11;

using namespace ec_pro_native;

PYBIND11_MODULE(
    _ec_pro_native,
    module
) {
    module.doc() =
        "EC Pro native inference, batching "
        "and token-processing layer";

    module.def(
        "version",
        []() {
            return "0.1.0";
        }
    );

    py::class_<EngineConfig>(
        module,
        "EngineConfig"
    )
        .def(py::init<>())
        .def_readwrite(
            "model_path",
            &EngineConfig::model_path
        )
        .def_readwrite(
            "n_ctx",
            &EngineConfig::n_ctx
        )
        .def_readwrite(
            "n_batch",
            &EngineConfig::n_batch
        )
        .def_readwrite(
            "n_threads",
            &EngineConfig::n_threads
        )
        .def_readwrite(
            "n_threads_batch",
            &EngineConfig::
                n_threads_batch
        )
        .def_readwrite(
            "n_gpu_layers",
            &EngineConfig::n_gpu_layers
        );

    py::class_<GenerationConfig>(
        module,
        "GenerationConfig"
    )
        .def(py::init<>())
        .def_readwrite(
            "max_new_tokens",
            &GenerationConfig::
                max_new_tokens
        )
        .def_readwrite(
            "temperature",
            &GenerationConfig::
                temperature
        )
        .def_readwrite(
            "top_p",
            &GenerationConfig::top_p
        )
        .def_readwrite(
            "seed",
            &GenerationConfig::seed
        );

    py::class_<ChatMessage>(
        module,
        "ChatMessage"
    )
        .def(py::init<>())
        .def_readwrite(
            "role",
            &ChatMessage::role
        )
        .def_readwrite(
            "content",
            &ChatMessage::content
        );

    py::class_<GenerationRequest>(
        module,
        "GenerationRequest"
    )
        .def(py::init<>())
        .def_readwrite(
            "messages",
            &GenerationRequest::messages
        )
        .def_readwrite(
            "generation",
            &GenerationRequest::generation
        );

    py::class_<GenerationResult>(
        module,
        "GenerationResult"
    )
        .def_readonly(
            "text",
            &GenerationResult::text
        )
        .def_readonly(
            "latency_ms",
            &GenerationResult::latency_ms
        )
        .def_readonly(
            "prompt_tokens",
            &GenerationResult::
                prompt_tokens
        )
        .def_readonly(
            "generated_tokens",
            &GenerationResult::
                generated_tokens
        );

    py::class_<
        InferenceEngine,
        std::shared_ptr<InferenceEngine>
    >(
        module,
        "InferenceEngine"
    )
        .def(
            py::init<EngineConfig>()
        )
        .def(
            "render_chat",
            &InferenceEngine::render_chat
        )
        .def(
            "generate",
            &InferenceEngine::generate,
            py::call_guard<
                py::gil_scoped_release
            >()
        )
        .def(
            "generate_batch",
            &InferenceEngine::
                generate_batch,
            py::arg("requests"),
            py::arg("parallelism") = 1,
            py::call_guard<
                py::gil_scoped_release
            >()
        )
        .def(
            "tokenize",
            &InferenceEngine::tokenize,
            py::call_guard<
                py::gil_scoped_release
            >()
        )
        .def(
            "tokenize_many",
            &InferenceEngine::
                tokenize_many,
            py::arg("texts"),
            py::arg("worker_count") = 0,
            py::call_guard<
                py::gil_scoped_release
            >()
        );

    py::class_<BatchScheduler>(
        module,
        "BatchScheduler"
    )
        .def(
            py::init(
                [](
                    std::shared_ptr<
                        InferenceEngine
                    > engine,
                    std::size_t max_batch_size,
                    std::size_t max_queue_size,
                    std::int64_t batch_wait_ms,
                    std::size_t batch_parallelism
                ) {
                    if (!engine) {
                        throw std::invalid_argument(
                            "InferenceEngine "
                            "cannot be null"
                        );
                    }

                    std::shared_ptr<
                        GenerationBackend
                    > backend = engine;

                    return std::make_unique<
                        BatchScheduler
                    >(
                        std::move(backend),
                        max_batch_size,
                        max_queue_size,
                        batch_wait_ms,
                        batch_parallelism
                    );
                }
            ),
            py::arg("engine"),
            py::arg(
                "max_batch_size"
            ) = 4,
            py::arg(
                "max_queue_size"
            ) = 64,
            py::arg(
                "batch_wait_ms"
            ) = 4,
            py::arg(
                "batch_parallelism"
            ) = 1
        )
        .def(
            "generate",
            &BatchScheduler::generate,
            py::call_guard<
                py::gil_scoped_release
            >()
        )
        .def(
            "queue_depth",
            &BatchScheduler::
                queue_depth
        )
        .def(
            "close",
            &BatchScheduler::close,
            py::call_guard<
                py::gil_scoped_release
            >()
    );
}