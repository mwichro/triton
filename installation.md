# Triton Development Installation Guide

This document outlines the steps to compile and install the development version of Triton into a Python virtual environment, including workarounds for known build issues.

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.8+
- CMake 3.20+
- Ninja
- A C++17 compatible compiler (GCC 9+ or Clang)
- CUDA Toolkit (if targeting NVIDIA GPUs)

## Installation Steps

1. **Create and Activate a Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r python/requirements.txt
   ```

3. **Compile and Install Triton (Editable Mode)**
   Due to a known issue with the GSan runtime build failing to find system headers (`climits`), use the following command to bypass it:
   ```bash
   TRITON_GSAN_CLANGXX=/usr/bin/true pip install -e .
   ```
   *Note: Using `/usr/bin/true` as the compiler effectively skips the GSan assembly step.*

## Frequently Asked Questions

### Can I install this into other environments?
Yes. Since this is an **editable installation** (`pip install -e .`), the virtual environment contains a link to the source code in this directory. To use it in another environment:
1. Activate the target environment.
2. Run the same `pip install -e .` command (with the GSan workaround) from this directory.
3. **Note:** Any changes you make to the source code will be reflected in *all* environments linked to this directory.

### What is the issue with `GSan`?
`GSan` (Grand Sanitizer) is an experimental tool for memory sanitization in Triton kernels. The build failure (`fatal error: 'climits' file not found`) occurs during the compilation of the GSan device-side library using `clang++` because it cannot locate standard C++ headers on your system.

**Does it impact GPU performance?**
**No.** GSan is a debugging and safety tool. Disabling it or skipping its build only means you cannot use the experimental memory sanitization features. It has **no impact** on the performance of generated GPU kernels during normal execution.

### Clang vs. GCC for Compilation
While `clang++` was used for the GSan component, the main Triton compiler and its MLIR infrastructure are typically built using the system's default C++ compiler (likely GCC in your case).

**Does this cause performance losses on GPU code?**
**No.** There are two types of "performance" here:
1. **Compilation Speed:** Triton's own compilation time might vary slightly depending on whether its C++ components (the compiler itself) were built with GCC or Clang, but this is usually negligible.
2. **GPU Kernel Performance:** This is the most important part. The performance of the code running on your GPU is determined by Triton's internal optimization passes and the **LLVM backend** (which Triton downloads and uses internally). The host compiler (GCC vs. Clang) used to build the Triton shared library does not affect the PTX or SASS code generated for the GPU.
