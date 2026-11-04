# KS Shadowing

Detects when chaotic trajectories of the Kuramoto-Sivashinsky (KS) equation closely follow unstable Relative Periodic Orbits (RPOs). An RPO returns to a spatially shifted copy of itself after one period: `u(x, t + T) = u(x + shift, t)`.

Two detection methods are implemented:

- **State Space Approach (SSA)**: L2 distance in physical space with spatial shift optimization.
- **Persistent Homology Approach (PHA)**: shift-invariant Wasserstein distance between persistence diagrams.

The [examples gallery](https://traviscasey.github.io/ks-shadowing/) shows both methods on committed result files.

## Installation

Requires Python 3.12-3.14, [uv](https://docs.astral.sh/uv/), and:

```bash
sudo apt-get install cmake pkg-config libfftw3-dev libeigen3-dev libboost-dev  # Debian/Ubuntu
brew install cmake pkg-config fftw eigen boost                                 # macOS

git clone https://github.com/TravisCasey/ks-shadowing.git
cd ks-shadowing
uv sync
```

## Usage

```bash
# Generate a trajectory and run SSA on it
uv run ks-detect --method ssa --trajectory-steps 5000 --downsample 23 \
    --min-duration 7 --resolution 256 --output results/demo/ssa.h5

# Run PHA on the same trajectory, with a delay embedding
uv run ks-detect --method pha --trajectory results/demo/trajectory.h5 --delay 17 \
    --min-duration 7 --resolution 256 --output results/demo/pha.h5
```

The trajectory is saved as `trajectory.h5` next to `--output`, and an existing one is never overwritten, so a later run in the same directory must reuse it with `--trajectory`.

```python
from ks_shadowing import load_results

metadata, trajectory, events = load_results("results/demo/pha.h5")
```

`ks-detect --help` lists all options. RPOs for domain size L=22 are in `data/`: `rpos_all.npz` (239) and `rpos_selected.npz` (the 16 used by default).

## Citation

To cite the code behind a result, use the version DOI of that release, listed on [Zenodo](https://doi.org/10.5281/zenodo.21695318). That link is the concept DOI, which always resolves to the newest release; use it only to refer to the project as a whole. `CITATION.cff` has the full metadata.

## Third-party code

`csrc/hera/vendor/` holds a subset of [Hera](https://github.com/anigmetov/hera) under the BSD 3-Clause License. See its `README.md` and `license.txt`.

## Development

```bash
uv sync --dev
uv sync --dev --reinstall-package ks-shadowing  # rebuild after changes under csrc/
uv run pytest
uv run ruff check . && uv run ruff format . && uv run ty check
uv run sphinx-build docs docs/_build/html
```
