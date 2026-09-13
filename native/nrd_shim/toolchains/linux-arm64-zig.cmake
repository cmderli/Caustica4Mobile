# CMake toolchain for cross-compiling the NRD shim to Linux arm64 (glibc) with Zig.
#
# One Zig tarball covers both the compiler and the target libraries: it ships clang, a libc++ built for the
# target, and glibc stubs for every version it supports. `zig c++` links that libc++ statically, so the
# resulting shared library depends on nothing but libc/libpthread/libdl, and pinning the target to glibc 2.17
# (the aarch64 baseline) keeps it loadable on every arm64 Linux, including distributions older than the
# compiler used here.
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=native/nrd_shim/toolchains/linux-arm64-zig.cmake \
#         -DZIG=/path/to/zig ...
# `zig` is taken from -DZIG, $ZIG, or PATH, in that order.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(ZIG "" CACHE FILEPATH "Path to the zig executable used as the cross compiler")
if(NOT ZIG)
  set(ZIG "$ENV{ZIG}")
endif()
if(NOT ZIG)
  find_program(ZIG zig)
endif()
if(NOT ZIG OR NOT EXISTS "${ZIG}")
  message(FATAL_ERROR "zig not found: pass -DZIG=<path> or set $ZIG")
endif()

# glibc version to bind against, as a Zig target suffix. 2.17 is the oldest glibc with aarch64 support, so
# requiring it costs nothing and buys compatibility with every arm64 Linux.
set(CAUSTICA_ZIG_GLIBC "2.17" CACHE STRING "glibc version the arm64 build targets")
set(_triple "aarch64-linux-gnu.${CAUSTICA_ZIG_GLIBC}")

# CMake invokes a compiler as one executable, while Zig needs a sub-command plus the target triple, so the
# three tools are wrapped once here rather than repeated in every flag variable.
set(_wrapper_dir "${CMAKE_BINARY_DIR}/zig-wrappers")
file(MAKE_DIRECTORY "${_wrapper_dir}")

function(caustica_zig_wrapper name subcommand)
  set(_path "${_wrapper_dir}/${name}")
  file(WRITE "${_path}" "#!/bin/sh\nexec \"${ZIG}\" ${subcommand} -target ${_triple} \"$@\"\n")
  file(CHMOD "${_path}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE)
  set(CMAKE_${name}_WRAPPER "${_path}" PARENT_SCOPE)
endfunction()

caustica_zig_wrapper(cc "cc")
caustica_zig_wrapper(cxx "c++")
set(CMAKE_C_COMPILER "${CMAKE_cc_WRAPPER}" CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER "${CMAKE_cxx_WRAPPER}" CACHE FILEPATH "" FORCE)

# The archiver sub-commands take no target triple (the objects it handles are already target objects).
set(_ar "${_wrapper_dir}/ar")
file(WRITE "${_ar}" "#!/bin/sh\nexec \"${ZIG}\" ar \"$@\"\n")
set(_ranlib "${_wrapper_dir}/ranlib")
file(WRITE "${_ranlib}" "#!/bin/sh\nexec \"${ZIG}\" ranlib \"$@\"\n")
file(CHMOD "${_ar}" "${_ranlib}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)
set(CMAKE_AR "${_ar}" CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB "${_ranlib}" CACHE FILEPATH "" FORCE)

# CMake's Ninja/Makefile generators pass "-Xlinker --dependency-file=<file>" to collect link dependencies.
# Zig's driver validates linker arguments and rejects that one, so the fallback (CMake's own dependency
# scanning) is used instead; only the granularity of incremental link dependencies changes.
set(CMAKE_LINK_DEPENDS_USE_LINKER OFF)
set(CMAKE_CXX_LINK_DEPENDS_USE_LINKER OFF)

# The target's headers and libraries all come from the Zig installation, so nothing is searched on the host.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
