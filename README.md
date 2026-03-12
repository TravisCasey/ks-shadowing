# KS Shadowing

Shadowing detection for chaotic trajectories of the Kuramoto-Sivashinsky (KS) equation.

This package detects when chaotic trajectories closely follow unstable Relative Periodic Orbits (RPOs). An RPO is a solution that returns to a spatially shifted copy of itself after one period: `u(x, t + T) = u(x - shift, t)`.

Two detection methods are implemented:

- **State Space Approach (SSA)**: L2 distance in physical space with spatial shift optimization
- **Persistent Homology Approach (PHA)**: Shift-invariant wasserstein distance between persistence diagrams

## Installation

Requires Python 3.12+, CMake, and the following system libraries:

- **FFTW3** - Fast Fourier Transform (for KS integrator)
- **Eigen3** - Linear algebra (for KS integrator)
- **Boost** - Headers only (for Hera Wasserstein distance)

Clone with submodules and install:
```bash
git clone --recurse-submodules https://github.com/TravisCasey/ks-shadowing.git
cd ks-shadowing
uv sync
```

If you already cloned without `--recurse-submodules`, initialize them with:
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

The package exposes two CLI entry points via `uv run`:

```bash
# Detect events with SSA
uv run ks-detect --method ssa --trajectory-steps 50000 --resolution 64

# Detect events with PHA
uv run ks-detect --method pha --trajectory-steps 20000 --resolution 32 --delay 4

# Use explicit threshold instead of threshold_quantile
uv run ks-detect --method ssa --trajectory-steps 50000 --resolution 64 --threshold 1.0

# Visualize the best event from a saved result file
uv run ks-visualize --input results/shadowing_results_ssa.h5
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Linting and formatting
uv run ruff check .
uv run ruff format .

# Type checking
uv run ty check

# Clear build cache if C++ library fails to build after a clean checkout
uv cache clean ks-shadowing
```
