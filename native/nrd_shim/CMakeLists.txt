cmake_minimum_required(VERSION 3.22)
project(caustica_nrd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

get_filename_component(NRD_SHIM_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(NRD_SHIM_OUTPUT_ROOT "${NRD_SHIM_REPO_ROOT}/build/native/nrd_shim"
    CACHE PATH "Directory for built NRD shim artifacts")
set(NRD_SHIM_OUTPUT_DIR "${NRD_SHIM_OUTPUT_ROOT}/$<LOWER_CASE:$<CONFIG>>")

function(nrdshim_set_artifact_dirs target)
  set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${NRD_SHIM_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${NRD_SHIM_OUTPUT_DIR}"
      ARCHIVE_OUTPUT_DIRECTORY "${NRD_SHIM_OUTPUT_DIR}")
endfunction()

# ---------------------------------------------------------------------------
# Vulkan headers
#
# The shim only needs the headers: every Vulkan entry point is obtained at
# runtime through the PFN_vkGetDeviceProcAddr passed in by the caller, so the
# shared library does not link against libvulkan.
# ---------------------------------------------------------------------------
set(NRD_SHIM_VULKAN_INCLUDE_DIR "" CACHE PATH
    "Directory containing vulkan/vulkan.h (overrides VULKAN_SDK and the fetched headers)")

if(NOT NRD_SHIM_VULKAN_INCLUDE_DIR)
  find_package(Vulkan QUIET)
  if(Vulkan_FOUND AND Vulkan_INCLUDE_DIRS)
    set(NRD_SHIM_VULKAN_INCLUDE_DIR "${Vulkan_INCLUDE_DIRS}")
  endif()
endif()

if(NOT NRD_SHIM_VULKAN_INCLUDE_DIR AND DEFINED ENV{VULKAN_SDK})
  if(EXISTS "$ENV{VULKAN_SDK}/include/vulkan/vulkan.h")
    set(NRD_SHIM_VULKAN_INCLUDE_DIR "$ENV{VULKAN_SDK}/include")
  endif()
endif()

if(NOT NRD_SHIM_VULKAN_INCLUDE_DIR)
  include(FetchContent)
  set(NRD_SHIM_VULKAN_HEADERS_TAG "v1.3.290" CACHE STRING "Vulkan-Headers tag to fetch")
  FetchContent_Declare(vulkan_headers
      GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
      GIT_TAG        ${NRD_SHIM_VULKAN_HEADERS_TAG}
      GIT_SHALLOW    TRUE
      SOURCE_DIR     "${CMAKE_CURRENT_BINARY_DIR}/_deps/vulkan-headers-src"
      BINARY_DIR     "${CMAKE_CURRENT_BINARY_DIR}/_deps/vulkan-headers-build")
  FetchContent_MakeAvailable(vulkan_headers)
  set(NRD_SHIM_VULKAN_INCLUDE_DIR "${vulkan_headers_SOURCE_DIR}/include")
endif()

if(NOT EXISTS "${NRD_SHIM_VULKAN_INCLUDE_DIR}/vulkan/vulkan.h")
  message(FATAL_ERROR "vulkan/vulkan.h not found in '${NRD_SHIM_VULKAN_INCLUDE_DIR}'")
endif()

# ---------------------------------------------------------------------------
# NVIDIA Real-time Denoisers (NRD)
#
# NRD is fetched from NVIDIA's own repository at configure time and is never
# vendored into this repository (see native/nrd_shim/README.md). The source,
# build, shader output and the fetched dependencies all live under this
# project's binary directory. For an offline build point the standard
# -DFETCHCONTENT_SOURCE_DIR_NRD=<dir> override at a pre-downloaded checkout.
#
# The SPIR-V shaders are embedded into the NRD static library; the shim reads
# them back through GetInstanceDesc()->pipelines[i].computeShaderSPIRV.
# A DXC with SPIR-V support must be reachable through DXC_SPIRV_PATH (and
# DXC_PATH), or through PATH. The compiler options NRD sets on its own target
# (-Wextra -Werror -fPIC -fvisibility=hidden) must not be touched.
# ---------------------------------------------------------------------------
set(NRD_STATIC_LIBRARY ON)           # link NRD into libcaustica_nrd.so
set(NRD_NRI OFF)                     # the shim talks raw Vulkan, not NRI
set(NRD_EMBEDS_SPIRV_SHADERS ON)     # mandatory: Vulkan pipelines need these blobs
set(NRD_EMBEDS_DXIL_SHADERS OFF)
set(NRD_EMBEDS_DXBC_SHADERS OFF)

# These are part of the ABI the shim was built against: the shaders compile the
# matching permutation and "LibraryDesc" does not expose these flags at runtime.
# The values below are NRD's defaults and match what nrd_shim.cpp assumes
# (no checkerboarding, no confidence inputs, no disocclusion-threshold mix,
# anti-firefly on, no viewport offset, no REBLUR performance mode).
set(NRD_SUPPORTS_CHECKERBOARD ON)
set(NRD_SUPPORTS_HISTORY_CONFIDENCE ON)
set(NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX ON)
set(NRD_SUPPORTS_ANTIFIREFLY ON)
set(NRD_SUPPORTS_VIEWPORT_OFFSET OFF)
set(REBLUR_PERFORMANCE_MODE OFF)

# "IN_NORMAL_ROUGHNESS" encoding. 0 = NormalEncoding::RGBA8_UNORM: the host
# writes the guides itself from Slang, and "rgba8" is the only format Slang can
# express for a storage image (it has no r10g10b10a2_unorm format string), so
# this avoids hand-packed "unknown"-format storage writes.
# 1 = RoughnessEncoding::LINEAR (NRD's recommended value; note that value 1 is
# LINEAR and 2 is SQRT_LINEAR). The shim verifies both at runtime against
# LibraryDesc and logs them.
set(NRD_NORMAL_ENCODING 0 CACHE STRING "" FORCE)
set(NRD_ROUGHNESS_ENCODING 1 CACHE STRING "" FORCE)

include(FetchContent)
FetchContent_Declare(nrd
    GIT_REPOSITORY https://github.com/NVIDIA-RTX/NRD.git
    GIT_TAG        v4.17.3            # commit 792eff196afdd350fd9c3f862119017ccb438a0e
    GIT_SHALLOW    TRUE
    SOURCE_DIR     "${CMAKE_CURRENT_BINARY_DIR}/_deps/nrd-src"
    BINARY_DIR     "${CMAKE_CURRENT_BINARY_DIR}/_deps/nrd-build")
FetchContent_MakeAvailable(nrd)

# ---------------------------------------------------------------------------
# Shim
# ---------------------------------------------------------------------------
add_library(caustica_nrd SHARED nrd_shim.cpp)

target_include_directories(caustica_nrd PRIVATE "${NRD_SHIM_VULKAN_INCLUDE_DIR}")
target_link_libraries(caustica_nrd PRIVATE NRD)

# The host's Slang shaders are written against these encodings, so the shim
# checks the values the linked library reports at runtime.
target_compile_definitions(caustica_nrd PRIVATE
    CAUSTICA_NRD_EXPECTED_NORMAL_ENCODING=${NRD_NORMAL_ENCODING}
    CAUSTICA_NRD_EXPECTED_ROUGHNESS_ENCODING=${NRD_ROUGHNESS_ENCODING})

# The shim is loaded by name from a JVM, so it may not have a single unresolved import: everything it calls
# has to be resolvable from its own DT_NEEDED list. This is also what proves the shim never references a
# Vulkan entry point directly (it reaches them through the device proc address the caller passes in).
if(UNIX AND NOT APPLE)
  target_link_options(caustica_nrd PRIVATE "-Wl,--no-undefined")
endif()

set_target_properties(caustica_nrd PROPERTIES
    OUTPUT_NAME "caustica_nrd"
    POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF
    # The three ABI entry points carry an explicit visibility("default")
    # attribute; everything else in this translation unit stays hidden.
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
nrdshim_set_artifact_dirs(caustica_nrd)

# NRD's static libraries are built without a visibility preset, so their symbols
# (NRD internals are hidden, but ShaderMakeBlob's are not) would otherwise land in
# this library's dynamic symbol table. Only the three ABI functions are exported.
#
# GNU ld and LLD do that with --exclude-libs, which hides everything pulled from an
# archive. Zig's compiler driver rejects the flag, so when the linker does not take it
# a version script pins the exported set instead - the same guarantee, stated directly.
if(UNIX AND NOT APPLE)
  include(CheckLinkerFlag)
  check_linker_flag(CXX "-Wl,--exclude-libs,ALL" CAUSTICA_NRD_HAS_EXCLUDE_LIBS)
  if(CAUSTICA_NRD_HAS_EXCLUDE_LIBS)
    target_link_options(caustica_nrd PRIVATE "-Wl,--exclude-libs,ALL")
  else()
    set(CAUSTICA_NRD_VERSION_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/caustica_nrd.map")
    file(WRITE "${CAUSTICA_NRD_VERSION_SCRIPT}"
        "CAUSTICA_NRD_1.0 {\n  global:\n"
        "    caustica_nrd_create;\n    caustica_nrd_denoise;\n    caustica_nrd_destroy;\n"
        "  local:\n    *;\n};\n")
    target_link_options(caustica_nrd PRIVATE "-Wl,--version-script=${CAUSTICA_NRD_VERSION_SCRIPT}")
    message(STATUS "caustica_nrd: linking with an export version script (no --exclude-libs support)")
  endif()
endif()
