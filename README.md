# KS Shadowing

Shadowing detection for chaotic trajectories of the Kuramoto-Sivashinsky equation.

Implements two methods for detecting when trajectories closely follow unstable Relative Periodic Orbits (RPOs):

- **State Space Approach (SSA)**: L2 distance with spatial shift optimization
- **Persistent Homology Approach (PHA)**: Wasserstein distance between persistence diagrams

The goal is to compare computational efficiency and validate that PHA achieves equivalent detection with reduced complexity.

## Status

In very early development.
