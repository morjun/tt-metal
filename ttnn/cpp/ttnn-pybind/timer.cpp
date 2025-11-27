// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/timer.hpp"

#include "ttnn/util/timer.hpp"

namespace ttnn::timer {

void py_module(py::module& module) {
    module.def(
        "get_duration", &ttnn::Timer::get_duration, py::arg("name"), "Get the duration of a timer in microseconds");
    module.def("reset_duration", &ttnn::Timer::reset_duration, py::arg("name"), "Reset the duration of a timer");
    module.def("reset_all", &ttnn::Timer::reset_all, "Reset all timer durations");
}

}  // namespace ttnn::timer
