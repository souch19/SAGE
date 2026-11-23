"""Coût des opérateurs post-hoc sur un split : FLOPs et latence du régresseur, latence de
compression par page, stockage, latence du MaxSim en ligne. --carbon mesure en plus le CO2
de l'indexation complète du split (CodeCarbon).

    python 05_analysis/cost.py --model colbert --split econ --budget 64abs
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage import methods as M
from sage.salience import load_profiles, load_regressor
from sage.utils import DEVICE, latency_ms, load_pickle, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--bench", default="vidore_v2", choices=list(C.BENCHMARKS))
    ap.add_argument("--split", default="econ")
    ap.add_argument("--budget", default="64abs")
    ap.add_argument("--methods", nargs="*", default=["paper"])
    ap.add_argument("--n-docs", type=int, default=16, help="pages pour FLOPs et latences")
    ap.add_argument("--kmeans", default="auto", choices=["auto", "sklearn", "gpu"])
    ap.add_argument("--carbon", action="store_true")
    ap.add_argument("--out", default=str(C.REPORTS))
    args = ap.parse_args()

    P = C.Paths(args.model)
    d = load_pickle(P.emb(args.bench, args.split))
    de, qe = d["doc_emb"], d["q_emb"]
    tag, k_of = C.parse_budget(args.budget)
    methods = [m for m in M.expand(args.methods) if m not in M.GROUPS["adaptive"]]
    reg, profiles = load_regressor(P, DEVICE), load_profiles(P)
    docs = [de[x].to(DEVICE).float() for x in list(de)[:args.n_docs]]
    kb = lambda n_vec: n_vec * P.dim * 2 / 1e3                      # fp16
    backend = lambda m: M.default_backend(m, P.dim) if args.kmeans == "auto" else args.kmeans

    flops = []
    for p in docs:
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            reg(p.unsqueeze(0))
        flops.append(fc.get_total_flops())
    with torch.no_grad():
        sal = [reg(p.unsqueeze(0)).squeeze(0) for p in docs]
    out = {"model": args.model, "split": args.split, "budget": tag, "n_docs": len(de),
           "native_vectors": float(np.mean([e.shape[0] for e in de.values()])),
           "saliency": {"mflops": np.mean(flops) / 1e6,
                        "ms": np.mean([latency_ms(lambda: reg(p.unsqueeze(0))) for p in docs])},
           "methods": {}}

    for m in methods:
        method = M.build(m, de, profiles, DEVICE, backend=backend(m))
        lat, vec = [], []
        for p, s in zip(docs, sal):
            k = k_of(p.shape[0])
            lat.append(latency_ms(lambda: method.compress(p, s, k)))
            vec.append(method.compress(p, s, k).shape[0])
        out["methods"][m] = {"compress_ms": float(np.mean(lat)), "vectors": float(np.mean(vec)),
                             "kb_per_page": kb(np.mean(vec))}

    queries = [qe[q].float().to(DEVICE) for q in list(d["rel"])[:8]]

    def maxsim_ms(n_vec, n_pages=128):
        pages = F.normalize(torch.randn(n_pages, int(round(n_vec)), P.dim, device=DEVICE), dim=-1)
        return latency_ms(lambda: [torch.einsum("td,nld->tnl", q, pages).max(-1).values.sum(0)
                                   for q in queries])
    n_budget = float(np.mean([k_of(e.shape[0]) for e in de.values()]))
    out["online_ms_per_128_pages"] = {"native": maxsim_ms(out["native_vectors"]), tag: maxsim_ms(n_budget)}

    if args.carbon:
        from codecarbon import EmissionsTracker
        for m in methods:
            method = M.build(m, de, profiles, DEVICE, backend=backend(m))
            tracker = EmissionsTracker(project_name=f"{args.model}_{m}", measure_power_secs=1,
                                       tracking_mode="process", log_level="error", output_dir=args.out)
            tracker.start()
            t0 = time.time()
            for e in de.values():
                p = e.to(DEVICE).float()
                with torch.no_grad():
                    method.compress(p, reg(p.unsqueeze(0)).squeeze(0), k_of(p.shape[0]))
            kg = tracker.stop()
            out["methods"][m].update(index_s=time.time() - t0, mg_co2_per_page=kg * 1e6 / len(de))

    print(f"{args.model} / {args.split} / {tag} : {len(de)} pages, "
          f"{out['native_vectors']:.0f} vecteurs/page ({kb(out['native_vectors']):.1f} Ko)")
    print(f"régresseur : {out['saliency']['mflops']:.2f} MFLOPs, {out['saliency']['ms']:.2f} ms/page")
    print(f"{'méthode':<12}{'ms/page':>10}{'vecteurs':>10}{'Ko/page':>10}{'mgCO2/page':>12}")
    for m, r in out["methods"].items():
        co2 = f"{r['mg_co2_per_page']:.3f}" if "mg_co2_per_page" in r else "-"
        print(f"{m:<12}{r['compress_ms']:>10.2f}{r['vectors']:>10.1f}{r['kb_per_page']:>10.1f}{co2:>12}")
    o = out["online_ms_per_128_pages"]
    print(f"MaxSim, 128 pages x {len(queries)} requêtes : natif {o['native']:.2f} ms, {tag} {o[tag]:.2f} ms "
          f"(x{o['native'] / o[tag]:.1f})")
    save_json(Path(args.out) / f"cost_{args.model}_{args.split}_{tag}.json", out)


if __name__ == "__main__":
    main()
