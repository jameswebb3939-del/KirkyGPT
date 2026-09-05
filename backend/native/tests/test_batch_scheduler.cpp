#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kirk_gpt_native/batch_scheduler.hpp"
#include "kirk_gpt_native/generation_backend.hpp"
#include "kirk_gpt_native/types.hpp"

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

GenerationRequest make_request(
    std::size_t id
) {
    GenerationRequest request;

    ChatMessage message;

    message.role = "user";

    message.content =
        "request-"
        + std::to_string(id);

    request.messages.push_back(
        std::move(message)
    );

    request.generation.max_new_tokens =
        16;

    request.generation.temperature =
        0.0F;

    request.generation.top_p =
        1.0F;

    return request;
}

GenerationResult make_result(
    const GenerationRequest& request
) {
    GenerationResult result;

    if (!request.messages.empty()) {
        result.text =
            request.messages.back()
                .content;
    }

    result.latency_ms = 1;
    result.prompt_tokens = 1;
    result.generated_tokens = 1;

    return result;
}


class RecordingBackend final
    : public GenerationBackend {
public:
    explicit RecordingBackend(
        std::chrono::milliseconds
            delay =
                std::chrono::milliseconds(
                    0
                )
    )
        : delay_(delay) {
    }

    std::vector<GenerationResult>
    generate_batch(
        const std::vector<
            GenerationRequest
        >& requests,
        std::size_t parallelism
    ) const override {
        if (
            delay_
            > std::chrono::milliseconds(
                0
            )
        ) {
            std::this_thread::sleep_for(
                delay_
            );
        }

        {
            std::lock_guard lock(
                mutex_
            );

            batch_sizes_.push_back(
                requests.size()
            );

            parallelism_.push_back(
                parallelism
            );

            total_requests_ +=
                requests.size();
        }

        std::vector<
            GenerationResult
        > results;

        results.reserve(
            requests.size()
        );

        for (
            const auto& request
            : requests
        ) {
            results.push_back(
                make_result(request)
            );
        }

        return results;
    }

    std::vector<std::size_t>
    batch_sizes() const {
        std::lock_guard lock(
            mutex_
        );

        return batch_sizes_;
    }

    std::vector<std::size_t>
    parallelism_values() const {
        std::lock_guard lock(
            mutex_
        );

        return parallelism_;
    }

    std::size_t
    total_requests() const {
        std::lock_guard lock(
            mutex_
        );

        return total_requests_;
    }

private:
    std::chrono::milliseconds
        delay_;

    mutable std::mutex mutex_;

    mutable std::vector<
        std::size_t
    > batch_sizes_;

    mutable std::vector<
        std::size_t
    > parallelism_;

    mutable std::size_t
        total_requests_ = 0;
};


class BlockingBackend final
    : public GenerationBackend {
public:
    std::vector<GenerationResult>
    generate_batch(
        const std::vector<
            GenerationRequest
        >& requests,
        std::size_t
    ) const override {
        {
            std::lock_guard lock(
                mutex_
            );

            started_ = true;
        }

        cv_.notify_all();

        {
            std::unique_lock lock(
                mutex_
            );

            cv_.wait(
                lock,
                [this]() {
                    return released_;
                }
            );
        }

        std::vector<
            GenerationResult
        > results;

        results.reserve(
            requests.size()
        );

        for (
            const auto& request
            : requests
        ) {
            results.push_back(
                make_result(request)
            );
        }

        return results;
    }

    bool wait_until_started(
        std::chrono::milliseconds
            timeout
    ) const {
        std::unique_lock lock(
            mutex_
        );

        return cv_.wait_for(
            lock,
            timeout,
            [this]() {
                return started_;
            }
        );
    }

    void release() const {
        {
            std::lock_guard lock(
                mutex_
            );

            released_ = true;
        }

        cv_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;

    mutable bool started_ = false;
    mutable bool released_ = false;
};


class FailingBackend final
    : public GenerationBackend {
public:
    std::vector<GenerationResult>
    generate_batch(
        const std::vector<
            GenerationRequest
        >&,
        std::size_t
    ) const override {
        throw std::runtime_error(
            "dummy backend failure"
        );
    }
};


class WrongCountBackend final
    : public GenerationBackend {
public:
    std::vector<GenerationResult>
    generate_batch(
        const std::vector<
            GenerationRequest
        >&,
        std::size_t
    ) const override {
        return {};
    }
};


void test_batch_collection() {
    auto backend =
        std::make_shared<
            RecordingBackend
        >();

    BatchScheduler scheduler(
        backend,
        4,
        64,
        100,
        2
    );

    std::vector<
        std::future<GenerationResult>
    > futures;

    for (
        std::size_t id = 0;
        id < 4;
        ++id
    ) {
        futures.push_back(
            scheduler.submit(
                make_request(id)
            )
        );
    }

    for (
        std::size_t id = 0;
        id < futures.size();
        ++id
    ) {
        const auto result =
            futures[id].get();

        require(
            result.text
                == "request-"
                    + std::to_string(
                        id
                    ),
            "Batch result was "
            "associated with the "
            "wrong request"
        );
    }

    scheduler.close();

    const auto batches =
        backend->batch_sizes();

    require(
        !batches.empty(),
        "No batch was recorded"
    );

    require(
        batches.front() == 4,
        "Expected first batch "
        "to contain 4 requests"
    );

    const auto parallelism =
        backend
            ->parallelism_values();

    require(
        !parallelism.empty()
            && parallelism.front()
                == 2,
        "batch_parallelism was "
        "not forwarded"
    );
}


void test_bounded_queue() {
    auto backend =
        std::make_shared<
            BlockingBackend
        >();

    BatchScheduler scheduler(
        backend,
        1,
        2,
        0,
        1
    );

    auto first =
        scheduler.submit(
            make_request(0)
        );

    require(
        backend->wait_until_started(
            std::chrono::seconds(2)
        ),
        "Backend did not start "
        "within timeout"
    );

    auto second =
        scheduler.submit(
            make_request(1)
        );

    auto third =
        scheduler.submit(
            make_request(2)
        );

    bool rejected = false;

    try {
        auto ignored =
            scheduler.submit(
                make_request(3)
            );

        (void) ignored;
    }
    catch (
        const std::runtime_error&
            error
    ) {
        rejected =
            std::string(
                error.what()
            ).find(
                "queue is full"
            ) != std::string::npos;
    }

    require(
        rejected,
        "Scheduler did not reject "
        "submission when queue "
        "was full"
    );

    backend->release();

    require(
        first.get().text
            == "request-0",
        "First request failed"
    );

    require(
        second.get().text
            == "request-1",
        "Second request failed"
    );

    require(
        third.get().text
            == "request-2",
        "Third request failed"
    );

    scheduler.close();
}


void test_exception_propagation() {
    auto backend =
        std::make_shared<
            FailingBackend
        >();

    BatchScheduler scheduler(
        backend,
        1,
        16,
        0,
        1
    );

    auto future =
        scheduler.submit(
            make_request(0)
        );

    bool propagated = false;

    try {
        (void) future.get();
    }
    catch (
        const std::runtime_error&
            error
    ) {
        propagated =
            std::string(
                error.what()
            ).find(
                "dummy backend failure"
            ) != std::string::npos;
    }

    require(
        propagated,
        "Backend exception was "
        "not propagated through "
        "the future"
    );

    scheduler.close();
}


void test_wrong_result_count() {
    auto backend =
        std::make_shared<
            WrongCountBackend
        >();

    BatchScheduler scheduler(
        backend,
        1,
        16,
        0,
        1
    );

    auto future =
        scheduler.submit(
            make_request(0)
        );

    bool rejected = false;

    try {
        (void) future.get();
    }
    catch (
        const std::runtime_error&
            error
    ) {
        rejected =
            std::string(
                error.what()
            ).find(
                "unexpected result count"
            ) != std::string::npos;
    }

    require(
        rejected,
        "Scheduler accepted an "
        "invalid backend result count"
    );

    scheduler.close();
}


void test_close_drains_queue() {
    auto backend =
        std::make_shared<
            RecordingBackend
        >(
            std::chrono::
                milliseconds(5)
        );

    BatchScheduler scheduler(
        backend,
        4,
        64,
        2,
        1
    );

    std::vector<
        std::future<GenerationResult>
    > futures;

    for (
        std::size_t id = 0;
        id < 12;
        ++id
    ) {
        futures.push_back(
            scheduler.submit(
                make_request(id)
            )
        );
    }

    scheduler.close();

    for (
        std::size_t id = 0;
        id < futures.size();
        ++id
    ) {
        const auto result =
            futures[id].get();

        require(
            result.text
                == "request-"
                    + std::to_string(
                        id
                    ),
            "Pending request was "
            "not drained correctly"
        );
    }

    require(
        backend->total_requests()
            == 12,
        "close() did not drain "
        "all queued requests"
    );
}


void test_submit_after_close() {
    auto backend =
        std::make_shared<
            RecordingBackend
        >();

    BatchScheduler scheduler(
        backend
    );

    scheduler.close();

    bool rejected = false;

    try {
        auto ignored =
            scheduler.submit(
                make_request(0)
            );

        (void) ignored;
    }
    catch (
        const std::runtime_error&
            error
    ) {
        rejected =
            std::string(
                error.what()
            ).find(
                "closed"
            ) != std::string::npos;
    }

    require(
        rejected,
        "Scheduler accepted work "
        "after close()"
    );
}


void test_concurrent_submissions() {
    constexpr std::size_t
        producer_count = 8;

    constexpr std::size_t
        requests_per_producer = 8;

    constexpr std::size_t
        expected_total =
            producer_count
            * requests_per_producer;

    auto backend =
        std::make_shared<
            RecordingBackend
        >(
            std::chrono::
                milliseconds(1)
        );

    BatchScheduler scheduler(
        backend,
        8,
        128,
        2,
        2
    );

    std::atomic<std::size_t>
        successful{0};

    std::mutex error_mutex;
    std::exception_ptr error;

    std::vector<std::thread>
        producers;

    producers.reserve(
        producer_count
    );

    for (
        std::size_t producer = 0;
        producer < producer_count;
        ++producer
    ) {
        producers.emplace_back(
            [
                producer,
                requests_per_producer,
                &scheduler,
                &successful,
                &error,
                &error_mutex
            ]() {
                for (
                    std::size_t offset = 0;
                    offset
                        < requests_per_producer;
                    ++offset
                ) {
                    const auto id =
                        producer
                            * requests_per_producer
                        + offset;

                    try {
                        auto future =
                            scheduler.submit(
                                make_request(
                                    id
                                )
                            );

                        const auto result =
                            future.get();

                        if (
                            result.text
                            != "request-"
                                + std::to_string(
                                    id
                                )
                        ) {
                            throw std::runtime_error(
                                "Concurrent result "
                                "association failed"
                            );
                        }

                        ++successful;
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

    for (auto& producer : producers) {
        producer.join();
    }

    scheduler.close();

    if (error) {
        std::rethrow_exception(
            error
        );
    }

    require(
        successful
            == expected_total,
        "Not all concurrent "
        "requests completed"
    );

    require(
        backend->total_requests()
            == expected_total,
        "Backend did not receive "
        "all concurrent requests"
    );
}


struct TestCase {
    const char* name;
    void (*function)();
};

}  // namespace


int main() {
    const std::vector<TestCase>
        tests = {
            {
                "batch collection",
                test_batch_collection,
            },
            {
                "bounded queue",
                test_bounded_queue,
            },
            {
                "exception propagation",
                test_exception_propagation,
            },
            {
                "wrong result count",
                test_wrong_result_count,
            },
            {
                "close drains queue",
                test_close_drains_queue,
            },
            {
                "submit after close",
                test_submit_after_close,
            },
            {
                "concurrent submissions",
                test_concurrent_submissions,
            },
        };

    std::size_t passed = 0;

    for (const auto& test : tests) {
        try {
            test.function();

            ++passed;

            std::cout
                << "[PASS] "
                << test.name
                << '\n';
        }
        catch (
            const std::exception&
                error
        ) {
            std::cerr
                << "[FAIL] "
                << test.name
                << ": "
                << error.what()
                << '\n';

            return 1;
        }
    }

    std::cout
        << passed
        << "/"
        << tests.size()
        << " native scheduler tests passed"
        << '\n';

    return 0;
}