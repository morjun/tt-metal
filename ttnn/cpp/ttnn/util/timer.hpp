#pragma once

#include <chrono>
#include <string>
#include <iostream>

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
        std::cout << "[Timer] " << name_ << ": " << duration << " us" << std::endl;
        stopped_ = true;
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    bool stopped_ = false;
};

}  // namespace ttnn
