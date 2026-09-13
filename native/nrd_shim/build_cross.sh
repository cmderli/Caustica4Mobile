#!/usr/bin/env bash
# Cross-compiles the NRD shim for arm64 targets and stages each one under
# build/native/nrd_shim/<platform>/release/.
#
#   ./build_cross.sh              # every target this host can build
#   ./build_cross.sh android      # arm64 Android (arm64-v8a), needs the NDK
#   ./build_cross.sh linux        # arm64 Linux (glibc), needs Zig
#
# Both targets reuse the host build's fetched sources (NRD, MathLib, ShaderMake, Vulkan-Headers) and the host
# ShaderMake + DXC that generate NRD's SPIR-V blobs, so run the ordinary host build once first (see README):
# the shaders NRD embeds are target-independent, and only the shader *compilers* are host tools.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../.. && pwd)"
HOST_BUILD="$ROOT/build/native/nrd_shim/build"
DEPS="$HOST_BUILD/_deps"
SHADERMAKE="$DEPS/shadermake-build/build/ShaderMake"
GENERATOR=Ninja
command -v ninja >/dev/null 2>&1 || GENERATOR="Unix Makefiles"

TARGETS="${1:-all}"
case "$TARGETS" in
    all) WANT_ANDROID=1; WANT_LINUX=1 ;;
    android) WANT_ANDROID=1; WANT_LINUX=0 ;;
    linux) WANT_ANDROID=0; WANT_LINUX=1 ;;
    *) echo "usage: $0 [all|android|linux]" >&2; exit 2 ;;
esac

# --- shared prerequisites ---------------------------------------------------
for dir in nrd-src mathlib-src shadermake-src vulkan-headers-src sse2neon-src; do
    if [ ! -d "$DEPS/$dir" ]; then
        echo "missing $DEPS/$dir; run the host build first (README, 'Build' section)" >&2
        exit 1
    fi
done
if [ ! -x "$SHADERMAKE" ]; then
    echo "missing $SHADERMAKE; run the host build first" >&2
    exit 1
fi

# llvm-objcopy is a host x86_64 utility that edits any ELF, so one tool strips both artifacts.
OBJCOPY="${OBJCOPY:-$(command -v llvm-objcopy || true)}"
if [ -z "$OBJCOPY" ]; then
    OBJCOPY="$(ls "${ANDROID_NDK_HOME:-/opt/android-ndk}"/toolchains/llvm/prebuilt/*/bin/llvm-objcopy 2>/dev/null | head -1)"
fi

# The host DXC compiles NRD's shaders to SPIR-V; it is not a target tool.
DXC="${DXC_PATH:-}"
if [ -z "$DXC" ]; then
    for candidate in "$(command -v dxc || true)" "$ROOT"/build/tools/dxc/bin/dxc /tmp/armcross/linux_dxc_*/bin/dxc; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then DXC="$candidate"; break; fi
    done
fi
if [ -z "$DXC" ]; then
    echo "DXC not found: set DXC_PATH to a host dxc with SPIR-V support" >&2
    exit 1
fi

common_args=(
    "-DCMAKE_BUILD_TYPE=Release"
    "-DNRD_SHIM_VULKAN_INCLUDE_DIR=$DEPS/vulkan-headers-src/include"
    "-DFETCHCONTENT_SOURCE_DIR_NRD=$DEPS/nrd-src"
    "-DFETCHCONTENT_SOURCE_DIR_MATHLIB=$DEPS/mathlib-src"
    "-DFETCHCONTENT_SOURCE_DIR_SHADERMAKE=$DEPS/shadermake-src"
    # MathLib pulls sse2neon (an SSE-to-NEON header shim) for every ARM target; a pinned checkout keeps the
    # cross builds offline and independent of that repository's default branch.
    "-DFETCHCONTENT_SOURCE_DIR_SSE2NEON=$DEPS/sse2neon-src"
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    "-DSHADERMAKE_PATH=$SHADERMAKE"
    "-DDXC_PATH=$DXC"
    "-DDXC_SPIRV_PATH=$DXC"
    # ShaderMake resolves its own compiler paths when these are empty, which would silently pick whatever
    # dxc happens to be on PATH instead of the one this build was told to use.
    "-DSHADERMAKE_DXC_PATH=$DXC"
    "-DSHADERMAKE_DXC_VK_PATH=$DXC"
)

# --- android arm64 ----------------------------------------------------------
build_android() {
    local ndk="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
    if [ -z "$ndk" ]; then
        for candidate in "$HOME"/Android/Sdk/ndk/* /opt/android-ndk*; do
            [ -d "$candidate" ] && ndk="$candidate"
        done
    fi
    if [ -z "$ndk" ] || [ ! -f "$ndk/build/cmake/android.toolchain.cmake" ]; then
        echo "Android NDK not found: set ANDROID_NDK_HOME" >&2
        return 1
    fi
    local api="${ANDROID_API:-26}"
    local build_dir="$ROOT/build/native/nrd_shim/build-android-arm64"
    echo "[android-arm64] NDK $ndk, API $api, arm64-v8a, c++_static"
    cmake -S . -B "$build_dir" -G "$GENERATOR" \
        "${common_args[@]}" \
        "-DNRD_SHIM_OUTPUT_ROOT=$ROOT/build/native/nrd_shim/android-arm64" \
        -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM="android-$api" \
        -DANDROID_STL=c++_static
    cmake --build "$build_dir" --parallel
    strip_debug "$ROOT/build/native/nrd_shim/android-arm64/release/libcaustica_nrd.so"
    report "$ROOT/build/native/nrd_shim/android-arm64/release/libcaustica_nrd.so"
}

# --- linux arm64 ------------------------------------------------------------
build_linux() {
    local zig="${ZIG:-$(command -v zig || true)}"
    if [ -z "$zig" ]; then
        echo "zig not found: install it or set ZIG=/path/to/zig (see README)" >&2
        return 1
    fi
    local build_dir="$ROOT/build/native/nrd_shim/build-linux-arm64"
    echo "[linux-arm64] zig $("$zig" version)"
    cmake -S . -B "$build_dir" -G "$GENERATOR" \
        "${common_args[@]}" \
        "-DNRD_SHIM_OUTPUT_ROOT=$ROOT/build/native/nrd_shim/linux-arm64" \
        "-DCMAKE_TOOLCHAIN_FILE=$PWD/toolchains/linux-arm64-zig.cmake" \
        "-DZIG=$zig"
    cmake --build "$build_dir" --parallel
    strip_debug "$ROOT/build/native/nrd_shim/linux-arm64/release/libcaustica_nrd.so"
    report "$ROOT/build/native/nrd_shim/linux-arm64/release/libcaustica_nrd.so"
}

# Drops the DWARF sections the cross toolchains emit by default, which halves the artifact without touching
# the dynamic symbol table the JVM resolves the three entry points through. KEEP_DEBUG_INFO=1 keeps them for
# on-device crash analysis.
strip_debug() {
    local artifact="$1"
    if [ -n "${KEEP_DEBUG_INFO:-}" ]; then
        return 0
    fi
    if [ -z "$OBJCOPY" ]; then
        echo "warning: no llvm-objcopy found; $artifact keeps its debug info" >&2
        return 0
    fi
    "$OBJCOPY" --strip-debug "$artifact" || echo "warning: could not strip $artifact" >&2
}

# Prints what the artifact is, what it needs at runtime, and which ABI it exports.
report() {
    local artifact="$1"
    [ -f "$artifact" ] || { echo "expected artifact missing: $artifact" >&2; return 1; }
    echo
    echo "--- $artifact"
    file -b "$artifact"
    echo -n "  runtime deps: "
    readelf -d "$artifact" | grep NEEDED | sed 's/.*\[\(.*\)\]/\1/' | tr '\n' ' '
    echo
    echo -n "  glibc floor:  "
    readelf --version-info "$artifact" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -u -V | tail -1
    echo "  ABI exports:  $(nm -D --defined-only "$artifact" | grep -c ' T caustica_nrd_') of 3"
    ls -la "$artifact" | awk '{printf "  size:         %.1f MiB\n", $5/1048576}'
}

status=0
[ "$WANT_ANDROID" = 1 ] && { build_android || status=1; }
[ "$WANT_LINUX" = 1 ] && { build_linux || status=1; }
exit "$status"
