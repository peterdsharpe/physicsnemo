# Layered variable-coefficient context pilot

This directory preserves the underpowered 500-step signal check for the
registered comparison of optical summaries and ordered coefficient context.

The pilot is not a scientific verdict. All five reports completed with finite
metrics, the scalar features of each layer-order pair were exactly identical,
and the true paired response contrast was 4.0%. At this short budget, the
ordered models improved average in-distribution field error relative to the
fixed optical carrier, but recovered less than 2% of the held-out layer-order
contrast. The registered 4,000-step, five-seed comparison is required to decide
whether this is an optimization transient or a failure of the feed-forward
context representation.

The composed transfer matrices accumulated a maximum determinant error of
`1.87e-10`; independent constant-profile and boundary identities remained at
`1.52e-13` and `2.84e-14`. The formal reducer therefore uses a determinant
tolerance of `1e-8` while retaining `1e-10` for the two direct solution
identities.

- `pilot/` contains the five signal-check reports.
- `STATUS` records successful terminal completion.

