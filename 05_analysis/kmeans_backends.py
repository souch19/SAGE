"""k-means sklearn (CPU) contre k-means torch (GPU) : latence par page et rétention nDCG@5 sur
un split complet, pour une méthode à base de k-means.

    python 05_analysis/kmeans_backends.py --model colbert --split econ --budgets 64abs 05pct
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage import methods as M
from sage.retrieval import evaluate, native_ndcg
from sage.salience import load_profiles, load_regressor, predict
from sage.utils import DEVICE, latency_ms, load_pickle


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--bench", default="vidore_v2", choices=list(C.BENCHMARKS))
    ap.add_argument("--split", default="econ")
    ap.add_argument("--budgets", nargs="*", default=["64abs"])
    ap.add_argument("--method", default="wkmeans", choices=["wkmeans", "kmeans", "XR_wkmeans", "XP_wkmeans"])
    ap.add_argument("--n-speed", type=int, default=32, help="pages chronométrées")
    args = ap.parse_args()

    P = C.Paths(args.model)
    d = load_pickle(P.emb(args.bench, args.split))
    de, qe, rel = d["doc_emb"], d["q_emb"], d["rel"]
    nat = native_ndcg(P, args.bench, args.split, d, DEVICE)
    sal = predict(load_regressor(P, DEVICE), de, DEVICE)
    profiles = load_profiles(P) if args.method in M.USES_PROFILES else None
    timed = {x: de[x].to(DEVICE).float() for x in list(de)[:args.n_speed]}

    print(f"{args.model} / {args.split} / {args.method} : {len(de)} pages, natif {nat:.1f}")
    print(f"{'budget':<8}{'backend':<9}{'ms/page':>9}{'nDCG@5':>9}{'rétention':>11}")
    for tag, k_of in (C.parse_budget(b) for b in args.budgets):
        for backend in ("sklearn", "gpu"):
            method = M.build(args.method, de, profiles, DEVICE, backend=backend)
            speed = [latency_ms(lambda: method.compress(p, sal[x], k_of(p.shape[0])), 0, 1)
                     for x, p in timed.items()]
            comp = {}
            for x, e in de.items():
                p = e.to(DEVICE).float()
                comp[x] = method.compress(p, sal[x], k_of(p.shape[0])).cpu()
            nd = evaluate(qe, comp, rel, DEVICE)["ndcg5"]
            print(f"{tag:<8}{backend:<9}{np.mean(speed):>9.1f}{nd:>9.2f}{100 * nd / nat:>10.2f}%")


if __name__ == "__main__":
    main()
