"""Tableaux de résultats : nDCG@5 et rétention (100 x méthode / index complet) pour tous les
modèles présents sous $SAGE_ARTIFACTS. Écrit {banc}.md et results.csv dans --out.

    python 05_analysis/aggregate.py
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.utils import load_pickle

ORDER = ["select", "stride", "wkmeans", "XR_wkmeans", "XP_wkmeans", "kmeans", "agc",
         "ptm_mean", "ptm_wavg", "random", "bottomk_red", "norm_sel", "profile_sel",
         "profile_wkm", "dyn_wkmeans", "omni_select", "omni_merge", "topk_query"]
PAPER = {"select": "S-Top-K", "stride": "S-Group", "wkmeans": "Sage", "XR_wkmeans": "S-Xr",
         "XP_wkmeans": "S-Xp", "kmeans": "k-means uniforme", "agc": "AGC",
         "ptm_mean": "Prune-then-Merge", "topk_query": "Top-K query-aware (borne haute)"}


def budget_key(tag):
    return (tag.endswith("abs"), int(tag[:-3]))


def collect(root, seed):
    """Lignes (modèle, banc, split, budget, méthode, nDCG@5, natif)."""
    rows, sfx = [], "" if seed == C.DEFAULT_SEED else f"_s{seed}"
    for mdir in sorted(p for p in Path(root).iterdir() if (p / "results").is_dir()):
        for bdir in sorted(p for p in (mdir / "results").iterdir() if p.is_dir()):
            for sdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
                nat = sdir / "native.pkl"
                if not nat.exists():
                    continue
                native = load_pickle(nat)["ndcg5"]
                res_dir = mdir / f"results{sfx}" / bdir.name / sdir.name
                for f in sorted(res_dir.glob("*/*.pkl")):
                    rows.append((mdir.name, bdir.name, sdir.name, f.parent.name, f.stem,
                                 load_pickle(f)["ndcg5"], native))
    return rows


def table(header, lines):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(map(str, line)) + " |" for line in lines]


def methods_of(keys):
    return [m for m in ORDER if m in keys] + sorted(set(keys) - set(ORDER))


def report(bench, rows):
    splits = list(C.BENCHMARKS.get(bench, {})) or sorted({r[2] for r in rows})
    splits = [s for s in splits if any(r[2] == s for r in rows)]
    nd = {(r[0], r[2], r[3], r[4]): r[5] for r in rows}
    native = {(r[0], r[2]): r[6] for r in rows}
    models = sorted({r[0] for r in rows})
    tags = sorted({r[3] for r in rows}, key=budget_key)

    def retention(model, tag, method):
        vals = [100 * nd[model, s, tag, method] / native[model, s]
                for s in splits if (model, s, tag, method) in nd]
        return (np.mean(vals), len(vals) == len(splits)) if vals else (None, False)

    md = [f"# {bench}", "",
          "Rétention = 100 x nDCG@5(méthode) / nDCG@5(index complet), moyennée sur les splits ; "
          "* = splits manquants.", "",
          "Noms du papier : " + ", ".join(f"{k} = {v}" for k, v in PAPER.items()) + ".", "",
          "## Rétention moyenne (%)", ""]
    for tag in tags:
        lines = []
        for m in methods_of({r[4] for r in rows if r[3] == tag}):
            cells = []
            for model in models:
                v, full = retention(model, tag, m)
                cells.append("" if v is None else f"{v:.1f}" + ("" if full else "*"))
            lines.append([m] + cells)
        md += [f"### {tag}", ""] + table(["méthode"] + models, lines) + [""]

    for model in models:
        md += [f"## {model} : nDCG@5 (rétention %)", ""]
        nat_line = ["natif"] + [f"{native[model, s]:.1f}" if (model, s) in native else "" for s in splits]
        for tag in [t for t in tags if any(k[0] == model and k[2] == t for k in nd)]:
            lines = [nat_line + [""]]
            for m in methods_of({k[3] for k in nd if k[0] == model and k[2] == tag}):
                cells = [f"{nd[model, s, tag, m]:.1f} ({100 * nd[model, s, tag, m] / native[model, s]:.1f})"
                         if (model, s, tag, m) in nd else "" for s in splits]
                v, full = retention(model, tag, m)
                lines.append([m] + cells + [f"{v:.1f}" + ("" if full else "*")])
            md += [f"### {tag}", ""] + table(["méthode"] + splits + ["rét. moy."], lines) + [""]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(C.ARTIFACTS))
    ap.add_argument("--out", default=str(C.REPORTS))
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED, help="lit results_s{seed}")
    args = ap.parse_args()

    rows = collect(args.root, args.seed)
    if not rows:
        raise SystemExit(f"aucun résultat sous {args.root}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.seed == C.DEFAULT_SEED else f"_s{args.seed}"
    with open(out / f"results{sfx}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "bench", "split", "budget", "method", "ndcg5", "native", "retention"])
        for r in rows:
            w.writerow(list(r[:5]) + [f"{r[5]:.4f}", f"{r[6]:.4f}", f"{100 * r[5] / r[6]:.4f}"])
    by_bench = defaultdict(list)
    for r in rows:
        by_bench[r[1]].append(r)
    for bench, rs in by_bench.items():
        (out / f"{bench}{sfx}.md").write_text(report(bench, rs))
        print(f"{out / f'{bench}{sfx}.md'} : {len({r[0] for r in rs})} modèles, {len(rs)} résultats")
    print(f"{out / f'results{sfx}.csv'}")


if __name__ == "__main__":
    main()
