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
        stopped_ = true;
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    bool stopped_ = false;
};

}  // namespace ttnn
