"""Pareto matrix of the ISLA kernel study: step time and peak memory for every option
in kernel_options.OPTIONS at tokens in {10k, 40k}, batch in {1, 2}, bf16 autocast and
fp32 (TF32 off, PyTorch's default), plus GeoTransolver (n_hidden 256, 12 layers, 256
slices, no local features: geotransolver_surface.yaml) eager and compiled as the
comparison anchor. One GPU, one process, sequential; OOM is recorded as a result.

Usage: python bench_pareto.py <out.json> [option,option,...] [--configs N:B:prec,...]
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_common as bc  # noqa: E402
from kernel_options import OPTIONS, apply_option  # noqa: E402

GT_KW = dict(out_dim=4, functional_dim=6, geometry_dim=3, global_dim=3, n_layers=12, dropout=0.0, n_head=8, act="gelu",
             mlp_ratio=4, slice_num=256, use_te=False, plus=False, include_local_features=False,
             radii=[0.1, 0.5, 2.0], neighbors_in_radius=[16, 32, 64], n_hidden_local=32, state_mixing_mode="weighted", n_hidden=256)


class GTWrap(torch.nn.Module):
    """GeoTransolver with ISLA's keyword interface: local embedding = [points, normals]."""

    def __init__(self):
        super().__init__()
        from physicsnemo.models.geotransolver import GeoTransolver
        torch.manual_seed(0)
        self.m = GeoTransolver(**GT_KW)

    def forward(self, points, normals, drive, measure_weights):
        return self.m(local_embedding=torch.cat([points, normals], dim=-1), local_positions=points,
                      global_embedding=drive[:, None, :], geometry=points)


DEFAULT_CONFIGS = [(10_000, 1, "bf16"), (10_000, 1, "fp32"), (10_000, 2, "bf16"), (10_000, 2, "fp32"),
                   (40_000, 1, "bf16"), (40_000, 1, "fp32"), (40_000, 2, "bf16"), (40_000, 2, "fp32")]


def main():
    out_path = sys.argv[1]
    names = list(OPTIONS)
    configs = DEFAULT_CONFIGS
    args = sys.argv[2:]
    if args and not args[0].startswith("--"):
        names = args[0].split(",")
        args = args[1:]
    if args and args[0] == "--configs":
        configs = [(int(c.split(":")[0]), int(c.split(":")[1]), c.split(":")[2]) for c in args[1].split(",")]
    res = dict(env=bc.env_info(), gt_kw=GT_KW, reference=bc.REF_KW, options={k: v["description"] for k, v in OPTIONS.items()}, runs=[])
    for n, batch, prec in configs:
        for name in names:
            torch._dynamo.reset()
            if name == "geotransolver_eager":
                build = lambda: GTWrap().cuda()  # noqa: E731
            elif name == "geotransolver_compiled":
                build = lambda: torch.compile(GTWrap().cuda())  # noqa: E731
            else:
                opt = OPTIONS[name]
                build = lambda opt=opt: apply_option(bc.make_model(**opt.get("model_kw", {}))[0], opt)  # noqa: E731
            rec = bc.run_config(build, n, batch, prec == "bf16")
            rec["option"] = name
            res["runs"].append(rec)
            print(f"{name:32s} N={n:6d} B={batch} {prec}: {rec.get('step_ms_median', float('nan')):8.1f} ms  peak {rec.get('peak_allocated_gib', float('nan')):5.2f} GiB  {rec['status']} {rec.get('error', '')[-200:] if rec['status'] != 'ok' else ''}", flush=True)
            bc.dump(res, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
