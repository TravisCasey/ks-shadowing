# KS Shadowing

Shadowing detection for chaotic trajectories of the Kuramoto-Sivashinsky equation.

Implements two methods for detecting when trajectories closely follow unstable Relative Periodic Orbits (RPOs):

- **State Space Approach (SSA)**: L2 distance with spatial shift optimization
- **Persistent Homology Approach (PHA)**: Wasserstein distance between persistence diagrams

## Installation

Requires Python 3.12+, CMake, FFTW3, and Eigen3.

```bash
# Install package
uv sync
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
```

Note: After a clean checkout, run `uv cache clean ks-shadowing` if the C++ library fails to build.
