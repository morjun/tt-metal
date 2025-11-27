#pragma once

#include <chrono>
#include <string>
#include <tt-logger/tt-logger.hpp>

namespace ttnn {

class Timer {
public:
    Timer(const std::string& name) : name_(name) { start_ = std::chrono::high_resolution_clock::now(); }

    ~Timer() { stop(); }

    void stop() {
        if (stopped_) {
            return;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        log_info(tt::LogMetal, "[Timer] {}: {} us", name_, duration);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            durations_[name_] += duration;
        }
        stopped_ = true;
    }

    static int64_t get_duration(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (durations_.find(name) != durations_.end()) {
            return durations_[name];
        }
        return -1;
    }

    static void reset_duration(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        durations_.erase(name);
    }

    static void reset_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        durations_.clear();
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    bool stopped_ = false;
    static std::map<std::string, int64_t> durations_;
    static std::mutex mutex_;
};

// Define static members
inline std::map<std::string, int64_t> Timer::durations_;
inline std::mutex Timer::mutex_;

}  // namespace ttnn
