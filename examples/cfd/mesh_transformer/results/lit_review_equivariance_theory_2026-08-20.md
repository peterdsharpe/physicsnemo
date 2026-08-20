# Literature review: symmetry & sample efficiency (agent report, 2026-08-20)

Key verdict on our G1 pattern (tie at n=14, GeoTransolver pulls ahead with a
steeper slope): theory predicts the DECAY of the equivariance advantage
(Elesedy & Zaidi 2021: gap = sigma^2 dim(A)/(n-d-1); Tahmasebi & Jegelka
2023: |G| sample-multiplier / mild exponent gain) but FORBIDS the sign flip
under well-specified symmetry and comparable hypothesis classes. Two
literature-supplied mechanisms produce exactly our measured pattern:

1. SYMMETRY MISSPECIFICATION (Petrache & Trivedi 2023): exact
   scale-equivariance is physically wrong (Reynolds dependence); exact
   rotation equivariance is wrong unless every symmetry-breaking vector
   (inflow, ground) is an explicit transformed input. Over-symmetrization =
   bias floor that binds as n grows.
2. INVARIANT-FEATURIZATION EXPRESSIVITY CEILING (Joshi et al. 2023 GWL;
   Pozdnyakov et al. PRL 2020): invariant-only layers cannot propagate
   orientation across hops and truncated invariant sets provably confound
   distinct geometries. Villar et al. 2021: invariant features ARE universal
   for scalar targets IF the invariant set is complete.

Also: Brehmer et al. 2024 (equivariance at scale: equal exponents, 7x
prefactor, equivariant model with FEATURE FIELDS never overtaken); Gerken
et al. 2022 (augmentation never catches up on dense equivariant tasks);
Wang/Walters relaxed equivariance beats both extremes on fluids.

Ranked design implications: (1) carry equivariant feature fields internally
(GATr-style), scalarize at output only; (2) match the enforced group to the
TRUE symmetry — drop/condition exact scale equivariance (Re), make inflow/
ground explicit inputs, soft-relax the remainder; (3) audit invariant-set
completeness with degenerate-pair probes.

Both mechanisms are cheaply distinguishable: (1) residual-vs-scale/Re
correlation on existing checkpoints; (2) separating-pair probes / feature
completeness audit.

[Full citation list in the agent transcript; primary: arXiv 2102.10333,
2303.14269, 2410.23179, 2305.17592, 2106.06610, 2301.09308, PRL
10.1103/PhysRevLett.125.166001]
