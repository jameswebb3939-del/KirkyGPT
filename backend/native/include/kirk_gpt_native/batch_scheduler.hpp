#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

#include "kirk_gpt_native/generation_backend.hpp"
#include "kirk_gpt_native/types.hpp"

namespace kirk_gpt_native {

class BatchScheduler {
public:
    BatchScheduler(
        std::shared_ptr<GenerationBackend>
            backend,
        std::size_t max_batch_size = 4,
        std::size_t max_queue_size = 64,
        std::int64_t batch_wait_ms = 4,
        std::size_t batch_parallelism = 1
    );

    ~BatchScheduler();

    BatchScheduler(
        const BatchScheduler&
    ) = delete;

    BatchScheduler& operator=(
        const BatchScheduler&
    ) = delete;

    std::future<GenerationResult>
    submit(
        GenerationRequest request
    );

    GenerationResult generate(
        GenerationRequest request
    );

    std::size_t queue_depth() const;

    void close();

private:
    struct Job {
        GenerationRequest request;

        std::promise<GenerationResult>
            promise;
    };

    void worker_loop();

    std::shared_ptr<GenerationBackend>
        backend_;

    std::size_t max_batch_size_;
    std::size_t max_queue_size_;

    std::chrono::milliseconds
        batch_wait_;

    std::size_t batch_parallelism_;

    mutable std::mutex mutex_;

    std::condition_variable cv_;

    std::deque<
        std::shared_ptr<Job>
    > queue_;

    bool stopping_ = false;

    std::thread worker_;
};

}  // namespace kirk_gpt_native