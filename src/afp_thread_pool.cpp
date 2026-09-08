#include "../include/afp_thread_pool.hpp"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace AFP {

namespace {

/*
    Reentrancy depth per thread. While a thread executes pool work
    (as the caller running its own chunk, or as a worker running its
    assigned range), nested parallelFor calls from that thread run
    serially. This makes reentrant use deadlock-free by construction.
*/
thread_local int t_pool_depth = 0;

class PoolImpl {
public:
    static PoolImpl &instance()
    {
        static PoolImpl pool;
        return pool;
    }

    std::size_t workerCount() const { return workers_.size(); }

    /*
        Run fn over [begin, end) in contiguous chunks. The calling
        thread participates: it executes the trailing chunks while the
        workers handle the leading ones, then it waits for the workers.

        Concurrent calls from multiple threads are serialized by
        run_mutex_: each parallel section still uses the whole pool.
        Calls made from inside pool work (any thread) run serially on
        the calling thread instead of queueing, which both avoids
        deadlock and is the right behavior for nested parallelism.
    */
    void run(std::size_t begin, std::size_t end, std::size_t minimum_chunk,
             const std::function<void(std::size_t, std::size_t)> &fn)
    {
        if (begin >= end) return;

        if (t_pool_depth > 0) {
            fn(begin, end);
            return;
        }

        const std::size_t total = end - begin;
        const std::size_t participants = workers_.size() + 1;

        std::size_t chunk = (total + participants - 1) / participants;
        if (chunk < minimum_chunk) chunk = minimum_chunk;
        if (chunk == 0) chunk = 1;

        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        for (std::size_t start = begin; start < end; start += chunk) {
            ranges.emplace_back(start, std::min(start + chunk, end));
        }

        /*
            Hold run_mutex_ for the whole parallel section so concurrent
            callers serialize on the pool instead of racing the shared
            job state. The busy_ flag + condition variable wakes the
            next caller as soon as the section completes.
        */
        std::unique_lock<std::mutex> run_lock(run_mutex_);
        run_available_.wait(run_lock, [this] { return !busy_; });
        busy_ = true;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_fn_ = fn;
            for (std::size_t w = 0; w < workers_.size(); ++w) {
                ranges_[w] = w < ranges.size()
                    ? ranges[w]
                    : std::pair<std::size_t, std::size_t>{0, 0};
            }
            workers_done_ = 0;
            ++generation_;
        }

        work_ready_.notify_all();

        /*
            Calling thread executes the trailing chunks. Depth is raised
            around the user callback so nested parallelFor from the
            callback runs serially instead of deadlocking on run_mutex_.
        */
        {
            ++t_pool_depth;
            struct DepthGuard {
                int &d;
                ~DepthGuard() { --d; }
            } depth_guard{t_pool_depth};

            try {
                for (std::size_t r = workers_.size(); r < ranges.size(); ++r) {
                    fn(ranges[r].first, ranges[r].second);
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    busy_ = false;
                }
                run_available_.notify_all();
                waitForWorkers();
                throw;
            }
        }

        waitForWorkers();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            busy_ = false;
        }
        run_available_.notify_all();
    }

private:
    PoolImpl()
    {
        unsigned hardware = std::thread::hardware_concurrency();
        if (hardware == 0) hardware = 1;
        if (hardware > 1) --hardware; // calling thread is the last participant

        workers_.reserve(hardware);
        for (unsigned i = 0; i < hardware; ++i) {
            workers_.emplace_back([this, i] { workerLoop(i); });
        }
        ranges_.assign(workers_.size(), {0, 0});
    }

    ~PoolImpl()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        work_ready_.notify_all();
        for (std::thread &worker : workers_) {
            worker.join();
        }
    }

    void waitForWorkers()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        all_done_.wait(lock, [this] { return workers_done_ == workers_.size(); });
    }

    void workerLoop(std::size_t index)
    {
        std::size_t seen_generation = 0;

        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            work_ready_.wait(lock, [this, &seen_generation] {
                return generation_ != seen_generation || stop_;
            });

            if (stop_) return;

            seen_generation = generation_;
            const std::function<void(std::size_t, std::size_t)> fn = job_fn_;
            const std::pair<std::size_t, std::size_t> range = ranges_[index];
            lock.unlock();

            if (range.first < range.second && fn) {
                ++t_pool_depth;
                struct DepthGuard {
                    int &d;
                    ~DepthGuard() { --d; }
                } depth_guard{t_pool_depth};

                try {
                    fn(range.first, range.second);
                } catch (...) {
                    // Exceptions must not prevent completion signalling.
                }
            }

            lock.lock();
            ++workers_done_;
            if (workers_done_ == workers_.size()) {
                all_done_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::vector<std::pair<std::size_t, std::size_t>> ranges_;
    std::function<void(std::size_t, std::size_t)> job_fn_;
    std::mutex mutex_;
    std::mutex run_mutex_;
    std::condition_variable work_ready_;
    std::condition_variable all_done_;
    std::condition_variable run_available_;
    std::size_t generation_ = 0;
    std::size_t workers_done_ = 0;
    bool busy_ = false;
    bool stop_ = false;
};

} // namespace

void ThreadPool::parallelFor(
    std::size_t begin,
    std::size_t end,
    std::size_t minimum_chunk,
    const std::function<void(std::size_t, std::size_t)> &fn)
{
    PoolImpl::instance().run(begin, end, minimum_chunk, fn);
}

std::size_t ThreadPool::parallelism()
{
    return PoolImpl::instance().workerCount() + 1;
}

} // namespace AFP
