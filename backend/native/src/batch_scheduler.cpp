#include "kirk_gpt_native/batch_scheduler.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kirk_gpt_native {

BatchScheduler::BatchScheduler(
    std::shared_ptr<GenerationBackend>
        backend,
    std::size_t max_batch_size,
    std::size_t max_queue_size,
    std::int64_t batch_wait_ms,
    std::size_t batch_parallelism
)
    : backend_(std::move(backend)),
      max_batch_size_(
          max_batch_size
      ),
      max_queue_size_(
          max_queue_size
      ),
      batch_wait_(
          batch_wait_ms
      ),
      batch_parallelism_(
          batch_parallelism
      ) {
    if (!backend_) {
        throw std::invalid_argument(
            "BatchScheduler requires "
            "a generation backend"
        );
    }

    if (max_batch_size_ < 1) {
        throw std::invalid_argument(
            "max_batch_size must be >= 1"
        );
    }

    if (max_queue_size_ < 1) {
        throw std::invalid_argument(
            "max_queue_size must be >= 1"
        );
    }

    if (batch_wait_ms < 0) {
        throw std::invalid_argument(
            "batch_wait_ms cannot be negative"
        );
    }

    if (batch_parallelism_ < 1) {
        throw std::invalid_argument(
            "batch_parallelism must be >= 1"
        );
    }

    worker_ = std::thread(
        &BatchScheduler::worker_loop,
        this
    );
}

BatchScheduler::~BatchScheduler() {
    close();
}

std::future<GenerationResult>
BatchScheduler::submit(
    GenerationRequest request
) {
    auto job =
        std::make_shared<Job>();

    job->request =
        std::move(request);

    auto future =
        job->promise.get_future();

    {
        std::lock_guard lock(
            mutex_
        );

        if (stopping_) {
            throw std::runtime_error(
                "BatchScheduler is closed"
            );
        }

        if (
            queue_.size()
            >= max_queue_size_
        ) {
            throw std::runtime_error(
                "Native inference queue is full"
            );
        }

        queue_.push_back(
            std::move(job)
        );
    }

    cv_.notify_one();

    return future;
}

GenerationResult
BatchScheduler::generate(
    GenerationRequest request
) {
    auto future =
        submit(
            std::move(request)
        );

    return future.get();
}

std::size_t
BatchScheduler::queue_depth() const {
    std::lock_guard lock(
        mutex_
    );

    return queue_.size();
}

void BatchScheduler::close() {
    {
        std::lock_guard lock(
            mutex_
        );

        if (stopping_) {
            return;
        }

        stopping_ = true;
    }

    cv_.notify_all();

    if (worker_.joinable()) {
        worker_.join();
    }
}

void BatchScheduler::worker_loop() {
    while (true) {
        std::vector<
            std::shared_ptr<Job>
        > jobs;

        {
            std::unique_lock lock(
                mutex_
            );

            cv_.wait(
                lock,
                [this]() {
                    return stopping_
                        || !queue_.empty();
                }
            );

            if (
                stopping_
                && queue_.empty()
            ) {
                return;
            }

            const auto deadline =
                std::chrono::
                    steady_clock::now()
                + batch_wait_;

            while (
                !stopping_
                && queue_.size()
                    < max_batch_size_
            ) {
                if (
                    cv_.wait_until(
                        lock,
                        deadline
                    )
                    == std::cv_status::
                        timeout
                ) {
                    break;
                }
            }

            const auto count =
                std::min(
                    max_batch_size_,
                    queue_.size()
                );

            jobs.reserve(count);

            for (
                std::size_t index = 0;
                index < count;
                ++index
            ) {
                jobs.push_back(
                    std::move(
                        queue_.front()
                    )
                );

                queue_.pop_front();
            }
        }

        std::vector<
            GenerationRequest
        > requests;

        requests.reserve(
            jobs.size()
        );

        for (const auto& job : jobs) {
            requests.push_back(
                job->request
            );
        }

        try {
            auto results =
                backend_->generate_batch(
                    requests,
                    batch_parallelism_
                );

            if (
                results.size()
                != jobs.size()
            ) {
                throw std::runtime_error(
                    "Generation backend returned "
                    "an unexpected result count"
                );
            }

            for (
                std::size_t index = 0;
                index < jobs.size();
                ++index
            ) {
                jobs[index]
                    ->promise
                    .set_value(
                        std::move(
                            results[index]
                        )
                    );
            }
        }
        catch (...) {
            const auto error =
                std::current_exception();

            for (auto& job : jobs) {
                job->promise
                    .set_exception(
                        error
                    );
            }
        }
    }
}

}  // namespace kirk_gpt_native