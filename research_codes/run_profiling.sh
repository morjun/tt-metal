#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Convenience script for running profiling benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$TT_METAL_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

show_help() {
    cat << EOF
TT-Metal Profiling Convenience Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  test              Test profiling setup
  quick             Quick profiling run (small config)
  standard          Standard profiling run (default config)
  full              Full analysis (high-level + low-level)
  sweep             Batch size sweep
  tracy             Run with Tracy profiling
  help              Show this help message

Options:
  --large-batch N   Large batch size
  --small-batch N   Small batch size
  --minibatches N   Number of minibatches
  --iterations N    Number of measurement iterations

Examples:
  $0 test                                  # Test setup
  $0 quick                                 # Quick run
  $0 standard                              # Standard run
  $0 full --iterations 10                  # Full analysis
  $0 sweep                                 # Batch size sweep
  $0 tracy                                 # With Tracy profiling

EOF
}

run_test() {
    print_header "Testing Profiling Setup"
    python research_codes/test_profiling_setup.py
}

run_quick() {
    print_header "Quick Profiling Run (Small Configuration)"
    print_info "Using: batch=128, features=2048, iterations=3"
    python research_codes/profiling_sharding_noc_python.py \
        --large-batch 128 \
        --small-batch 16 \
        --minibatches 8 \
        --in-features 2048 \
        --out-features 2048 \
        --warmup 1 \
        --iterations 3 \
        "$@"
}

run_standard() {
    print_header "Standard Profiling Run"
    print_info "Using: batch=256, features=4096, iterations=5"
    python research_codes/profiling_sharding_noc_python.py "$@"
}

run_full() {
    print_header "Full Analysis (High-Level + Low-Level)"
    print_info "This will run both benchmarks and generate combined report"
    python research_codes/run_complete_analysis.py "$@"
}

run_sweep() {
    print_header "Batch Size Sweep"
    print_info "Running profiling for batch sizes: 64, 128, 256, 512"

    for batch in 64 128 256 512; do
        small_batch=$((batch / 8))
        print_info "Running batch=$batch (minibatch=$small_batch x 8)..."
        python research_codes/profiling_sharding_noc_python.py \
            --large-batch $batch \
            --small-batch $small_batch \
            --minibatches 8 \
            --output "results_batch_${batch}.csv" \
            --iterations 5 \
            "$@"
    done

    print_success "Sweep complete! Results saved to results_batch_*.csv"
}

run_tracy() {
    print_header "Running with Tracy Profiling"

    # Check if profiler is enabled
    if [ -z "$TT_METAL_DEVICE_PROFILER" ]; then
        print_info "Setting TT_METAL_DEVICE_PROFILER=1"
        export TT_METAL_DEVICE_PROFILER=1
    fi

    # Check if tracy module is available
    if ! python -c "import tracy" 2>/dev/null; then
        print_error "Tracy module not available"
        print_info "Install with: pip install tracy"
        exit 1
    fi

    # Check if profiler was built
    if [ ! -f "$TT_METAL_ROOT/build_metal.sh" ]; then
        print_error "build_metal.sh not found"
        exit 1
    fi

    print_info "Running with Tracy profiling enabled..."
    python -m tracy -r research_codes/profiling_sharding_noc_python.py "$@"

    print_success "Tracy profiling complete!"
    print_info "Open the generated .tracy file with Tracy GUI to visualize results"
}

# Main script logic
COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    test)
        run_test "$@"
        ;;
    quick)
        run_quick "$@"
        ;;
    standard)
        run_standard "$@"
        ;;
    full)
        run_full "$@"
        ;;
    sweep)
        run_sweep "$@"
        ;;
    tracy)
        run_tracy "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo
        show_help
        exit 1
        ;;
esac
