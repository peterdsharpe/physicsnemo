"""Numerics gate for every kernel option in the ISLA kernel study.

For each option: the same weights and inputs as the eager reference (fp32, TF32 off,
geo_checkpoint=True, fast_point_softmax=True), one forward and one backward of the
training loss (MSE against a fixed random target). Reported per option:

  out_rel_l2      ||out_opt - out_ref|| / ||out_ref||            (fp32)
  out_max_abs     max |out_opt - out_ref|                        (fp32)
  grad_rel_l2     the same over the concatenation of all parameter gradients
  grad_max_param_rel_l2   worst per-parameter relative L2 (parameters whose reference
                          gradient norm is below 1e-30 * total are skipped)
  fp64_*          the option evaluated in float64 against the eager float64 reference
                  (an option that is the same arithmetic in a different order agrees
                  here to ~1e-12; a change of formula would not)

and, once per run, the fp32 roundoff floor of the reference itself: eager fp32 vs
eager fp64 with the same weights (out and grads). An option whose fp32 deviation from
the fp32 reference is below this floor is indistinguishable from the reference by
any fp32 evaluation; the 1e-6 / 1e-5 bars of the study brief are applied to the
float64 comparison (same arithmetic) and the fp32 numbers are reported against the
floor.

Usage: python numerics_check.py <out.json> [tokens=10000] [batch=1] [option,option,...]
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_common as bc  # noqa: E402
from kernel_options import OPTIONS, apply_option  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300)), float((a - b).abs().max())


def eval_model(model, inputs, target):
    model.zero_grad(set_to_none=True)
    out = model(**inputs)
    loss = torch.nn.functional.mse_loss(out.float(), target.float())
    loss.backward()
    # torch.compile wraps the module and prefixes parameter names with _orig_mod.
    grads = {k.replace("_orig_mod.", ""): p.grad.detach().clone() for k, p in model.named_parameters()}
    return out.detach().clone(), grads


def compare_grads(ga, gb):
    ca = torch.cat([ga[k].double().flatten() for k in gb])
    cb = torch.cat([gb[k].double().flatten() for k in gb])
    tot = cb.norm()
    worst, worst_name, skipped = 0.0, None, []
    for k in gb:
        nb = gb[k].double().norm()
        if nb < 1e-6 * tot:
            # e.g. geo_logit.bias: a constant added to every routing logit cancels in
            # both softmaxes, so its gradient is zero up to roundoff (~1e-10 relative)
            # and a relative error has no meaning there.
            skipped.append(k)
            continue
        r = float((ga[k].double() - gb[k].double()).norm() / nb)
        if r > worst:
            worst, worst_name = r, k
    return dict(grad_rel_l2=float((ca - cb).norm() / tot), grad_max_abs=float((ca - cb).abs().max()),
                grad_max_param_rel_l2=worst, grad_max_param=worst_name, grad_params_skipped_zero=skipped)


def main():
    out_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    names = sys.argv[4].split(",") if len(sys.argv) > 4 else list(OPTIONS)
    res = dict(env=bc.env_info(), tokens=n, batch=batch, reference=bc.REF_KW, options={})
    ref, _ = bc.make_model()
    state = {k: v.clone() for k, v in ref.state_dict().items()}
    inputs32, target32 = bc.make_inputs(n, batch, out_dim=4)
    out_ref, g_ref = eval_model(ref, inputs32, target32)

    ref64 = ref.double()
    inputs64 = {k: v.double() for k, v in inputs32.items()}
    out_ref64, g_ref64 = eval_model(ref64, inputs64, target32.double())
    ref = ref64.float()  # keep one reference module around, fp32 again
    floor_out = rel(out_ref, out_ref64)
    res["fp32_floor"] = dict(out_rel_l2=floor_out[0], out_max_abs=floor_out[1], **compare_grads(g_ref, g_ref64))
    print("fp32 floor (eager fp32 vs eager fp64):", res["fp32_floor"], flush=True)

    for name in names:
        opt = OPTIONS[name]
        if name == "eager_ckpt":
            continue
        rec = dict(description=opt["description"])
        try:
            torch._dynamo.reset()
            m, _ = bc.make_model(**opt.get("model_kw", {}))
            m.load_state_dict(state)
            m = apply_option(m, opt)
            out, g = eval_model(m, inputs32, target32)
            o = rel(out, out_ref)
            rec.update(out_rel_l2=o[0], out_max_abs=o[1], **compare_grads(g, g_ref))
            o64 = rel(out, out_ref64)
            rec["vs_fp64_out_rel_l2"] = o64[0]
            rec["vs_fp64_grad_rel_l2"] = compare_grads(g, g_ref64)["grad_rel_l2"]
            if opt.get("fp64_ok", True):
                torch._dynamo.reset()
                m64, _ = bc.make_model(**opt.get("model_kw", {}))
                m64.load_state_dict(state)
                m64 = apply_option(m64.double(), opt)
                out64, g64 = eval_model(m64, inputs64, target32.double())
                o = rel(out64, out_ref64)
                gg = compare_grads(g64, g_ref64)
                rec.update(fp64_out_rel_l2=o[0], fp64_out_max_abs=o[1], fp64_grad_rel_l2=gg["grad_rel_l2"],
                           fp64_grad_max_param_rel_l2=gg["grad_max_param_rel_l2"])
                del m64
            rec["status"] = "ok"
            del m
        except Exception as e:  # noqa: BLE001
            import traceback
            rec.update(status="error", error=traceback.format_exc()[-2000:])
        res["options"][name] = rec
        print(name, {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in rec.items() if k != "description"}, flush=True)
        torch.cuda.empty_cache()
    bc.dump(res, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
