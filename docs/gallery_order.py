"""Explicit narrative ordering for the Sphinx-Gallery examples.

Lives in its own module, rather than in ``conf.py``, so that ``conf.py`` can
reference it by fully qualified name. Sphinx caches the config by pickling it,
which a callable defined in ``conf.py`` cannot survive.
"""

EXAMPLE_ORDER = [
    "plot_shadowing_event.py",
    "plot_shadowing_matrices.py",
    "plot_shadowing_diagrams.py",
    "plot_event_histogram.py",
    "plot_matched_events.py",
    "plot_coverage_vs_embedding.py",
    "plot_derivative_spectrum.py",
    "plot_derivative_saturation.py",
    "plot_derivative_mechanism.py",
    "plot_derivative_rescaling.py",
    "plot_computational_cost.py",
]


def example_order(name: str) -> int:
    """Sort key for gallery scripts; unlisted scripts sort to the end."""
    return [*EXAMPLE_ORDER, name].index(name)
