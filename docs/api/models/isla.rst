ISLA
====

ISLA (Invariant Slice Attention) is an SE(3)-equivariant, measure-aware
slice-attention surrogate for steady boundary-value problems. A boundary sample
(positions, unit normals, quadrature measures, optional per-cell boundary
scalars) and the problem's global inputs (zero or more global vector inputs and
zero or more global scalar inputs) go in; fields on the boundary, or at interior
query points, come out. The attention operates only on invariants of the
per-point vector set, and the frame is re-attached at the vector heads, so
rotation and translation covariance holds by construction.

ISLA lives in ``physicsnemo.experimental.nn`` and may change between releases.

Contracts
---------

- **Rotation/translation equivariance.** The backbone sees only invariants;
  vector outputs are expanded in the input vector set plus its spherical-basis
  complements with invariant coefficients. With ``similarity_gauge=True`` the
  model is additionally equivariant to geometric scale.
- **Measure-aware aggregation.** Slice states are quadrature-weighted means, so
  the routing reads a sampled integral rather than a raw point population.
- **Query independence (optional).** With ``query_independent=True`` queries are
  decoded by passive read blocks and a prediction at one point does not depend
  on which other points are queried.

Every constructor argument and every ``forward`` input is keyword-only. The
reference configuration uses the relative frame (``frame_mode="relative"``) with
the measure-weighted RMS pairwise-distance scale
(``scale_mode="rms_distance"``), both class defaults:

.. code-block:: python

    model = ISLA(
        out_scalars=1,
        out_vectors=1,
        hidden=192,
        n_layers=12,
        n_slices=256,
        mlp_ratio=4,
        geo_checkpoint=True,
    )
    out = model(
        points=points,                  # (B, N, 3)
        normals=normals,                # (B, N, 3)
        measure_weights=measure,        # (B, N)
        global_vectors=direction,       # (B, K, 3), K = n_global_vectors
    )

Interior queries are passed as ``query_points`` (with ``query_normals`` and
``query_scalars``); ``query_tokens=True`` admits them as interacting tokens,
``query_independent=True`` decodes them passively, and ``support_tokens=True``
adds a per-case computational support (``support_points``, ``support_normals``,
``support_scalars``). Global scalar inputs are passed as ``global_scalars`` of
shape ``(B, S)`` with ``S = n_global_scalars``.

The unified external-aerodynamics recipe
(``examples/cfd/external_aerodynamics/unified_external_aero_recipe``) documents
the reference configuration, its variants, and the data-to-model mapping.

.. autoclass:: physicsnemo.experimental.nn.isla.model.ISLA
    :show-inheritance:
    :members:
    :exclude-members: forward
