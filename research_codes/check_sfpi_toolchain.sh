#!/usr/bin/env bash
# Audit the SFPI kernel toolchain across every git worktree.
#
# WHY THIS EXISTS. runtime/sfpi is the RISC-V compiler that JIT-compiles every
# Tensix kernel at run time, and its version is pinned per-commit in
# tt_metal/sfpi-version. Different branches legitimately pin different versions.
# If a worktree's runtime provides a version other than the one its own source
# pins, it still RUNS -- it just builds the kernels with the wrong compiler, and
# every device timing shifts silently. Measured on 12B: the ring-vs-mcast
# contrast moved -75.78 -> -80.63 us (-4.84) between SFPI 7.67.0 and 7.79.0 with
# the host build held constant. See gemma4-12b/MEASUREMENT_RECORD.md §6.7.
#
# THE INVARIANT: a worktree's runtime must PROVIDE exactly the version its own
# tt_metal/sfpi-version PINS. Sharing one runtime between worktrees is fine only
# when their pins agree -- it is the drift, not the sharing, that corrupts.
#
# Exit 0 if every worktree is consistent, 1 otherwise.
set -uo pipefail
rc=0
printf "%-34s %-9s %-9s %-7s %s\n" WORKTREE PINNED PROVIDES STATUS "runtime"
while read -r w; do
    [ -d "$w" ] || continue
    pin=$(sed -n "s/.*sfpi_version=['\"]\?\([0-9.]*\).*/\1/p" "$w/tt_metal/sfpi-version" 2>/dev/null | head -1)
    gpp="$w/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
    if [ -x "$gpp" ]; then
        prov=$("$gpp" --version 2>/dev/null | sed -n 's/.*sfpi:\([0-9.]*\).*/\1/p' | head -1)
    else
        prov="-"
    fi
    if [ -L "$w/runtime" ]; then tgt="-> $(readlink "$w/runtime")"; else tgt="(own)"; fi
    if [ -z "$pin" ] || [ "$prov" = "-" ]; then
        st="SKIP"
    elif [ "$pin" = "$prov" ]; then
        st="OK"
    else
        st="DRIFT"; rc=1
    fi
    printf "%-34s %-9s %-9s %-7s %s\n" "$(basename "$w")" "${pin:-?}" "$prov" "$st" "$tgt"
done < <(git worktree list --porcelain | awk '/^worktree /{print $2}')
[ $rc -eq 0 ] && echo "OK: every worktree provides its own pinned SFPI." \
              || echo "DRIFT: a worktree is compiling kernels with the wrong SFPI -- device timings from it are not comparable."
exit $rc
