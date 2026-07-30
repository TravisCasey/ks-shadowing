# KS Shadowing

Shadowing detection for chaotic trajectories of the Kuramoto-Sivashinsky (KS) equation.

This package detects when chaotic trajectories closely follow unstable Relative Periodic Orbits (RPOs). An RPO is a solution that returns to a spatially shifted copy of itself after one period: `u(x, t + T) = u(x + shift, t)`.

Two detection methods are implemented:

- **State Space Approach (SSA)**: L2 distance in physical space with spatial shift optimization.
- **Persistent Homology Approach (PHA)**: shift-invariant Wasserstein distance between persistence diagrams.

## Installation

Requires Python 3.12 or 3.13, CMake, and the following system libraries:

- **FFTW3** -- Fast Fourier Transform (for the KS integrator)
- **Eigen3** -- Linear algebra (for the KS integrator)
- **Boost** -- Headers only (for the Hera Wasserstein distance library)

Clone with submodules and install:
```bash
git clone --recurse-submodules https://github.com/TravisCasey/ks-shadowing.git
cd ks-shadowing
uv sync
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

## RPO Data

RPO data files are in the `data/` directory. All RPOs are for domain size L=22.

| File | Description |
|------|-------------|
| `rpos_all.npz` | Complete dataset of 239 RPOs |
| `rpos_selected.npz` | 16 RPOs selected for focused analysis |

## Command-Line Usage

```bash
# Detect events with SSA
uv run ks-detect --method ssa --trajectory-steps 50000 --resolution 64

# Detect events with PHA at delay 4
uv run ks-detect --method pha --trajectory-steps 20000 --resolution 32 --delay 4

# Manual threshold instead of quantile-based auto-detection
uv run ks-detect --method ssa --trajectory-steps 50000 --resolution 64 --threshold 1.0
```

See the sphinx-gallery examples directory for plotting and analysis of detected shadowing.

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Rebuild C++ extensions after changes under csrc/ (plain uv sync --dev does not rebuild)
uv sync --dev --reinstall-package ks-shadowing

# Run tests
uv run pytest

# Linting and formatting
uv run ruff check .
uv run ruff format .

# C++ style and lint configuration: .clang-format and .clang-tidy (project files
# under csrc/, excluding the csrc/hera/upstream submodule)

# Type checking
uv run ty check

# Build HTML docs (requires dev dependencies)
uv run sphinx-build docs docs/_build/html

# Clear build cache if the C++ library fails to build after a clean checkout
uv cache clean ks-shadowing
```
