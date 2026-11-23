"""Variabilité selon la graine (régresseur, profils, initialisations k-means) : rétention moyenne
de chaque méthode pour chaque graine, puis moyenne et écart-type entre graines.
La graine 42 correspond à results/, les autres à results_s{graine}/ (./run.sh seeds).

    python 05_analysis/seed_variance.py --model colqwen --seeds 42 1 2 3
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.utils import load_pickle


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--bench", default="vidore_v2", choices=list(C.BENCHMARKS))
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    args = ap.parse_args()

    splits = list(C.splits(args.bench))
    ret = defaultdict(dict)                      # (budget, méthode) -> {graine: rétention}
    for seed in args.seeds:
        P = C.Paths(args.model, seed)
        nat = {s: load_pickle(P.native(args.bench, s))["ndcg5"]
               for s in splits if P.native(args.bench, s).exists()}
        found = defaultdict(dict)
        for s in nat:
            for f in P.root.glob(f"results{P.sfx}/{args.bench}/{s}/*/*.pkl"):
                found[f.parent.name, f.stem][s] = 100 * load_pickle(f)["ndcg5"] / nat[s]
        for key, per_split in found.items():
            if len(per_split) == len(splits):
                ret[key][seed] = np.mean(list(per_split.values()))

    print("| budget | méthode | " + " | ".join(f"s{s}" for s in args.seeds) + " | moyenne | écart-type |")
    print("|" + "---|" * (len(args.seeds) + 4))
    for tag, m in sorted(ret, key=lambda k: (k[0].endswith("abs"), k[0], k[1])):
        vals = [ret[tag, m].get(s) for s in args.seeds]
        ok = [v for v in vals if v is not None]
        cells = " | ".join("" if v is None else f"{v:.2f}" for v in vals)
        std = f"{np.std(ok, ddof=1):.2f}" if len(ok) > 1 else ""
        print(f"| {tag} | {m} | {cells} | {np.mean(ok):.2f} | {std} |")


if __name__ == "__main__":
    main()
