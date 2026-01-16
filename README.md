# KS Shadowing

Shadowing detection for chaotic trajectories of the Kuramoto-Sivashinsky (KS) equation.

This package detects when chaotic trajectories closely follow unstable Relative Periodic Orbits (RPOs). An RPO is a solution that returns to a spatially shifted copy of itself after one period: `u(x, t + T) = u(x - shift, t)`.

Two detection methods are planned:

- **State Space Approach (SSA)**: L2 distance in physical space with spatial shift optimization (implemented)
- **Persistent Homology Approach (PHA)**: Wasserstein distance between persistence diagrams (planned)

## Installation

Requires Python 3.12+, CMake, FFTW3, and Eigen3.

```bash
uv sync
```

## RPO Data

RPO data files are in the `data/` directory. All RPOs are for domain size L=22.

| File | Description |
|------|-------------|
| `rpos_all.npz` | Complete dataset of 239 RPOs |
| `rpos_selected.npz` | 16 RPOs selected for focused analysis |

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
