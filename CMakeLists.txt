cmake_minimum_required(VERSION 3.17)
project(ks_shadowing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type: Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Build type (Release, Debug, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# Defaults to -march=x86-64-v3 on x86_64 (Haswell baseline: AVX2, FMA, BMI2;
# 2013+ hardware) and empty on other architectures. -march=native is avoided
# so binaries built on an HPC login node remain portable across compute
# nodes. Override with -DKS_TARGET_ARCH=... at configure time.
include(CheckCXXCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_ks_default_arch "-march=x86-64-v3")
else()
  set(_ks_default_arch "")
endif()
set(KS_TARGET_ARCH "${_ks_default_arch}" CACHE STRING
    "Architecture flag passed to the compiler (e.g. -march=x86-64-v3). Empty disables.")

add_library(project_options INTERFACE)
target_compile_options(project_options INTERFACE
    -O3 -Wall -Wextra -Wpedantic)
if(KS_TARGET_ARCH)
  unset(KS_TARGET_ARCH_OK CACHE)
  check_cxx_compiler_flag("${KS_TARGET_ARCH}" KS_TARGET_ARCH_OK)
  if(KS_TARGET_ARCH_OK)
    target_compile_options(project_options INTERFACE "${KS_TARGET_ARCH}")
  else()
    message(WARNING
        "Compiler rejected KS_TARGET_ARCH=${KS_TARGET_ARCH}; "
        "proceeding without an explicit -march flag.")
  endif()
endif()

add_subdirectory(csrc)
