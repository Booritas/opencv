// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024 Intel Corporation


#include "thread_pool.hpp"

#include <opencv2/gapi/util/throw.hpp>

ncvslideio::gapi::own::Latch::Latch(const uint64_t expected)
    : m_expected(expected) {
}

void ncvslideio::gapi::own::Latch::count_down() {
    std::lock_guard<std::mutex> lk{m_mutex};
    --m_expected;
    if (m_expected == 0) {
        m_all_done.notify_all();
    }
}

void ncvslideio::gapi::own::Latch::wait() {
    std::unique_lock<std::mutex> lk{m_mutex};
    while (m_expected != 0u) {
        m_all_done.wait(lk);
    }
}

ncvslideio::gapi::own::ThreadPool::ThreadPool(const uint32_t num_workers) {
    m_workers.reserve(num_workers);
    for (uint32_t i = 0; i < num_workers; ++i) {
        m_workers.emplace_back(
                ncvslideio::gapi::own::ThreadPool::worker, std::ref(m_queue));
    }
}

void ncvslideio::gapi::own::ThreadPool::worker(QueueClass<Task>& queue) {
    while (true) {
        ncvslideio::gapi::own::ThreadPool::Task task;
        queue.pop(task);
        if (!task) {
            break;
        }
        task();
    }
}

void ncvslideio::gapi::own::ThreadPool::schedule(ncvslideio::gapi::own::ThreadPool::Task&& task) {
    m_queue.push(std::move(task));
};

void ncvslideio::gapi::own::ThreadPool::shutdown() {
    for (size_t i = 0; i < m_workers.size(); ++i) {
        // NB: Empty task - is an indicator for workers to stop their loops
        m_queue.push({});
    }
    for (auto& worker : m_workers) {
        worker.join();
    }
    m_workers.clear();
}

ncvslideio::gapi::own::ThreadPool::~ThreadPool() {
    shutdown();
}
