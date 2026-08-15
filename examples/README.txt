Examples
========

A gallery of analysis and plotting scripts that consume the result files in
``examples/data/``. All scripts share a single trajectory
(``examples/data/trajectory.h5``) so cross-method and cross-parameter
comparisons are consistent.

Notation
--------

PHA has two embedding parameters, written throughout the gallery as:

- :math:`w`: the **delay window**: the number of consecutive Wasserstein
  distances averaged along the diagonal of the per-RPO distance matrix.
  :math:`w = 1` applies no delay embedding.
- :math:`\lambda`: the **number of spatial-derivative orders** averaged over.
  :math:`\lambda = 1` uses the field alone; :math:`\lambda = 3` averages orders
  0, 1 and 2.

Both stages average rather than sum, so a distance stays on the scale of a
single-snapshot Wasserstein value whatever the setting:

.. math::

   D_{w,\lambda}(i, j) = \frac{1}{w\lambda} \sum_{l=0}^{w-1}
   \sum_{m=0}^{\lambda-1} W_m\bigl(i + l,\ (j + l) \bmod J\bigr),

where :math:`W_m` is the order-:math:`m` Wasserstein matrix and :math:`J` the
RPO period.

These map onto the API and the fixture filenames as:

- :math:`w` is the ``delay`` parameter, and appears in filenames as ``d{w}``.
- :math:`\lambda` is ``max_derivative_order`` **plus one**, and appears in
  filenames as ``o{lambda - 1}``.

Note the offset: ``max_derivative_order`` is the **highest** order included,
while :math:`\lambda` **counts** the orders averaged over, so
:math:`\lambda =` ``max_derivative_order`` :math:`+\ 1`.

Figures that plot a quantity per individual derivative order, rather than per
embedding, label that axis "Derivative order" and index it from 0: it is an
order index, not a count.
