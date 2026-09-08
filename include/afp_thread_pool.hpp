#pragma once

#include <cstddef>
#include <functional>

namespace AFP {

/*
    Minimal fixed-size thread pool shared by all tensor-level parallel
    operations (matrix-vector and matrix multiply).

    Why a pool: spawning hardware_concurrency() threads inside every
    matrix operation costs tens of microseconds per call and, when the
    caller is already running images in parallel, oversubscribes the
    machine N^2-fold. A process-wide pool with the calling thread as a
    participant keeps parallelism inside the tensor operation while
    paying the thread-creation cost exactly once.

    Reentrant calls (a pool job that itself calls parallelFor) are
    detected and executed serially on the calling thread, so the pool
    can never deadlock on nested parallelism.
*/
class ThreadPool {
public:
    /*
        Run fn(begin, end) over [begin, end) in contiguous chunks.
        The calling thread participates as a worker; the call returns
        only when every chunk has completed. minimum_chunk bounds the
        smallest chunk size (0 means: split so every worker gets work).
    */
    static void parallelFor(
        std::size_t begin,
        std::size_t end,
        std::size_t minimum_chunk,
        const std::function<void(std::size_t, std::size_t)> &fn);

    // Total participants for parallelFor (pool workers + calling thread).
    static std::size_t parallelism();
};

} // namespace AFP
