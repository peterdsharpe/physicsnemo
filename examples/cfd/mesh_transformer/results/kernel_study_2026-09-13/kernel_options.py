"""Registry of the kernel-level options tried in the ISLA kernel study (2026-09-13).

Every option leaves the model's arithmetic unchanged (same operations up to
floating-point order); options differ in fusion, recompute strategy and shape
specialization. ``model_kw`` are constructor overrides on top of the reference
configuration; ``apply`` post-processes the built module (e.g. wraps it in
torch.compile). ``fp64_ok`` marks options that can also be evaluated in float64 for
the same-arithmetic check.
"""
import torch


def _compile_model(mode=None, dynamic=False, **kw):
    def apply(m):
        return torch.compile(m, mode=mode, dynamic=dynamic, **kw)
    return apply


def _compile_blocks(dynamic=False, mode=None):
    def apply(m):
        for b in m.blocks:
            b.forward = torch.compile(b.forward, dynamic=dynamic, mode=mode)
        return m
    return apply


def _identity(m):
    return m


OPTIONS = {
    "eager_ckpt": dict(description="reference: eager, geo_checkpoint=True (isla_surface_reference.yaml)", model_kw={}, apply=_identity),
    "eager_nockpt": dict(description="eager, geo_checkpoint=False (store the (B,N,S,.) geometry instead of recomputing it)",
                         model_kw=dict(geo_checkpoint=False), apply=_identity),
    "eager_ckpt_slow_softmax": dict(description="eager, geo_checkpoint=True, fast_point_softmax=False: the pre-2026-09-10 middle-dimension "
                                                "point softmax, the configuration behind the book's GB300 '3.7x' matched-memory number",
                                    model_kw=dict(fast_point_softmax=False), apply=_identity),
    "compile_model_ckpt": dict(description="torch.compile(model) (the recipe's compile: true path), geo_checkpoint=True",
                               model_kw={}, apply=_compile_model()),
    "compile_model_nockpt": dict(description="torch.compile(model), geo_checkpoint=False (Inductor's own min-cut recompute decides what is saved)",
                                 model_kw=dict(geo_checkpoint=False), apply=_compile_model()),
    "compile_blocks_ckpt": dict(description="torch.compile on each _SliceBlock.forward only (embed/head eager), geo_checkpoint=True",
                                model_kw={}, apply=_compile_blocks()),
    "compile_model_dynamic": dict(description="torch.compile(model, dynamic=True): one kernel set for every token count",
                                  model_kw=dict(geo_checkpoint=False), apply=_compile_model(dynamic=True)),
    "compile_model_maxautotune": dict(description="torch.compile(model, mode='max-autotune-no-cudagraphs'), geo_checkpoint=False",
                                      model_kw=dict(geo_checkpoint=False), apply=_compile_model(mode="max-autotune-no-cudagraphs")),
    "compile_model_cudagraphs": dict(description="torch.compile(model, mode='reduce-overhead') (CUDA graphs), geo_checkpoint=False",
                                     model_kw=dict(geo_checkpoint=False), apply=_compile_model(mode="reduce-overhead")),
    "fused_geo_eager": dict(description="geo_kernel='fused' (Triton fused geometry region, recompute built in), rest eager",
                            model_kw=dict(geo_kernel="fused"), apply=_identity),
    "fused_geo_compile_model": dict(description="geo_kernel='fused' + torch.compile(model) (fused region is an opaque custom op inside the compiled graph)",
                                    model_kw=dict(geo_kernel="fused"), apply=_compile_model()),
}


def apply_option(model, opt):
    return opt["apply"](model)
