"""Intervalles de confiance à 95 % de la rétention et comparaison appariée à une méthode de référence.

Bootstrap sur les requêtes : dans chaque split, les requêtes sont tirées avec remise, avec les
mêmes indices pour toutes les méthodes et pour l'index complet (tirage apparié). Rétention d'un
tirage = moyenne sur les splits de 100 x nDCG@5 moyen (méthode) / nDCG@5 moyen (natif).
Il faut le champ per_query dans les résultats (relancer evaluate.py --force sur d'anciens fichiers).

    python 05_analysis/significance.py --model colbert --bench vidore_v2 --budgets 64abs --ref agc
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.retrieval import evaluate
from sage.utils import DEVICE, load_pickle


def native_per_query(P, bench, split):
    nat = load_pickle(P.native(bench, split))
    if "per_query" in nat:
        return nat["per_query"]
    d = load_pickle(P.emb(bench, split))             # ancien native.pkl : recalcul en mémoire
    return evaluate(d["q_emb"], d["doc_emb"], d["rel"], DEVICE)["per_query"]


def bootstrap(per_split, n_boot, seed):
    """per_split : liste de (natif [n], {méthode: [n]}). Renvoie {méthode: rétentions [n_boot]}."""
    rng, out = np.random.default_rng(seed), {}
    for nat, meth in per_split:
        idx = rng.integers(0, len(nat), size=(n_boot, len(nat)))
        den = nat[idx].mean(1)
        for m, v in meth.items():
            out.setdefault(m, []).append(100 * v[idx].mean(1) / den)
    return {m: np.mean(v, axis=0) for m, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--bench", default="vidore_v2", choices=list(C.BENCHMARKS))
    ap.add_argument("--budgets", nargs="*", default=["64abs"])
    ap.add_argument("--methods", nargs="*", default=None, help="défaut : toutes celles présentes")
    ap.add_argument("--ref", default="wkmeans", help="méthode de référence des comparaisons")
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED, help="résultats de cette graine")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default=None, help="fichier Markdown de sortie")
    args = ap.parse_args()

    P = C.Paths(args.model, args.seed)
    splits = [s for s in C.splits(args.bench) if P.native(args.bench, s).exists()]
    md = [f"# {args.model} / {args.bench} : IC 95 % (bootstrap apparié, {args.n_boot} tirages)", ""]
    for tag in [C.parse_budget(b)[0] for b in args.budgets]:
        per_split, missing = [], set()
        for s in splits:
            files = {f.stem: f for f in P.result_dir(args.bench, s, tag).glob("*.pkl")}
            wanted = args.methods or sorted(files)
            pq = {m: load_pickle(files[m]).get("per_query") for m in wanted if m in files}
            missing |= {m for m in wanted if pq.get(m) is None}
            nat = native_per_query(P, args.bench, s)
            qids = list(nat)
            per_split.append((np.array([nat[q] for q in qids]),
                              {m: np.array([v[q] for q in qids]) for m, v in pq.items() if v is not None}))
        common = set.intersection(*(set(m) for _, m in per_split)) if per_split else set()
        if missing:
            print(f"[{tag}] sans per_query (relancer evaluate.py --force) : {', '.join(sorted(missing))}")
        if not common:
            continue
        boot = bootstrap([(n, {m: v[m] for m in common}) for n, v in per_split], args.n_boot, args.seed)
        point = {m: np.mean([100 * v[m].mean() / n.mean() for n, v in per_split]) for m in common}
        lines = []
        for m in sorted(common, key=lambda x: -point[x]):
            lo, hi = np.percentile(boot[m], [2.5, 97.5])
            row = f"| {m} | {point[m]:.1f} | [{lo:.1f}, {hi:.1f}] |"
            if args.ref in common and m != args.ref:
                diff = boot[m] - boot[args.ref]
                dlo, dhi = np.percentile(diff, [2.5, 97.5])
                p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
                row += f" {point[m] - point[args.ref]:+.1f} [{dlo:+.1f}, {dhi:+.1f}] | {p:.3f} |"
            else:
                row += " | |"
            lines.append(row)
        md += [f"## {tag} ({len(per_split)} splits)", "",
               f"| méthode | rétention % | IC 95 % | écart à {args.ref} [IC 95 %] | p |",
               "|---|---|---|---|---|"] + lines + [""]
    text = "\n".join(md)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
