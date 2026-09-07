Examples
========

A gallery of analysis and plotting scripts that consume the result files in
``examples/data/``. All scripts share a single trajectory
(``examples/data/trajectory.h5``) so cross-method and cross-parameter
comparisons are consistent.

Notation
--------

PHA compares length-:math:`m` sequences of persistence diagrams, obtained by
one of two embedding methods, written throughout the gallery as:

- ``PHA--DELAY``: the diagrams of :math:`m` consecutive trajectory timesteps.
- ``PHA--DERIV``: the diagrams of the spatial-derivative fields of orders
  :math:`0` to :math:`m - 1` of a single snapshot, where order 0 is the field
  itself.

:math:`m = 1` reduces both to plain ``PHA``. Either way the :math:`m`
Wasserstein distances are averaged rather than summed, so a distance stays on
the scale of a single-snapshot value whatever the setting. Writing
:math:`c = \lfloor (m-1)/2 \rfloor` for the window-centering offset, the delay
embedding averages along the diagonal of the per-RPO distance matrix,

.. math::

   W_m(i, j) = \frac{1}{m} \sum_{l=0}^{m-1}
   W\bigl(i + l - c,\ (j + l - c) \bmod J\bigr),

and the derivative embedding averages across the per-order matrices,

.. math::

   W_m(i, j) = \frac{1}{m} \sum_{l=0}^{m-1} W^{(l)}(i, j),

where :math:`W^{(l)}` is the order-:math:`l` Wasserstein matrix, :math:`J` the
RPO period, and :math:`T` the trajectory length in timesteps. The window mean
is attributed to its center, so the delay-embedded :math:`W_m(i, j)` is
defined for :math:`c \le i \le T - 1 - \lfloor m/2 \rfloor`. Matrix names
follow the paper: :math:`W` is the PHA distance matrix, :math:`D` the SSA one,
and :math:`d_{W^2}` the Wasserstein metric itself.

Each entry of :math:`W^{(l)}` is the :math:`d_{W^2}` distance between full
sublevel-set persistence diagrams. A diagram holds the finite :math:`H_0`
pairs plus two essential classes with infinite death: the component born as the
state minimum and the loop born at the field maximum. Infinite points cannot be
matched to the diagonal, so the essential classes of two diagrams pair with each
other at cost equal to their birth difference, and

.. math::

   d_{W^2}^2 = d_{\mathrm{fin}}^2 + (\min u - \min u')^2 + (\max u - \max u')^2,

where :math:`d_{\mathrm{fin}}` is the :math:`d_{W^2}` matching of the finite
pairs alone.

These map onto the API and the fixture filenames as:

- For ``PHA--DELAY``, :math:`m` is the ``delay`` parameter, and appears in
  filenames as ``d{m}``.
- For ``PHA--DERIV``, :math:`m` is ``max_derivative_order`` **plus one**, and
  appears in filenames as ``o{m - 1}``.

Note the offset: ``max_derivative_order`` is the **highest** order included,
while :math:`m` **counts** the orders averaged over, so
:math:`m =` ``max_derivative_order`` :math:`+\ 1`. The two embeddings compose
freely in the API (``delay`` and ``max_derivative_order`` are independent
parameters), but the gallery never combines them.

Figures that plot a quantity per individual derivative order, rather than per
embedding, label that axis "Derivative order" and index it from 0: it is an
order index, not a count. Axes over :math:`m` are labeled "Embedding
length".

Detection strategies are named as the paper's ``\texttt`` macros render them:
monospace ``SSA``, ``PHA`` (no embedding), ``PHA--DELAY`` (delay embedding) and
``PHA--DERIV`` (derivative embedding), where the dash renders as a single en
dash in figure text (written as the escape ``\u2013``). Figure text one-indexes
RPOs; the API, filenames and result files are zero-indexed.
