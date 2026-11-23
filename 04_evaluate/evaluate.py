"""Compresse l'index de chaque split avec chaque méthode et budget, puis mesure le nDCG@5.

Groupes de méthodes (sage/methods.py) : paper (défaut), main, ablation, trainfree, adaptive, all.
Sorties : results[_s{seed}]/{banc}/{split}/{budget}/{méthode}.pkl
          {ndcg5, recall, per_query, avg_vectors, compression, time, seed, backend}
          results/{banc}/{split}/native.pkl ; un résultat existant est sauté (sauf --force).

    python 04_evaluate/evaluate.py --model colqwen --bench vidore_v2
    python 04_evaluate/evaluate.py --model colbert --methods ablation trainfree --budgets 05pct 64abs
"""
import argparse
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage import methods as M
from sage.retrieval import evaluate, native_ndcg
from sage.salience import load_profiles, load_regressor, predict
from sage.utils import DEVICE, load_pickle, save_pickle


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--bench", default="vidore_v2", choices=list(C.BENCHMARKS))
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=["paper"], help="méthodes ou groupes")
    ap.add_argument("--budgets", nargs="*", default=C.BUDGETS, help="ex. 05pct 10pct 64abs")
    ap.add_argument("--kmeans", default="auto", choices=["auto", "sklearn", "gpu"],
                    help="backend k-means ; auto = réglage des résultats du papier")
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED,
                    help="graine des k-means et du tirage aléatoire ; choisit aussi régresseur et profils")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    P = C.Paths(args.model, args.seed)
    methods = M.expand(args.methods)
    budgets = [C.parse_budget(b) for b in args.budgets]
    predictor = load_regressor(P, DEVICE) if M.USES_SALIENCE & set(methods) else None
    profiles = load_profiles(P) if M.USES_PROFILES & set(methods) else None

    for split in C.splits(args.bench, args.splits):
        jobs = [(m, tag, k_of) for m in methods for tag, k_of in budgets
                if args.force or not P.result(args.bench, split, tag, m).exists()]
        if not jobs:
            print(f"[{args.model}/{split}] déjà évalué")
            continue
        if not P.emb(args.bench, split).exists():
            print(f"[{args.model}/{split}] embeddings absents ({P.emb(args.bench, split)})")
            continue
        d = load_pickle(P.emb(args.bench, split))
        de, qe, rel = d["doc_emb"], d["q_emb"], d["rel"]
        nat = native_ndcg(P, args.bench, split, d, DEVICE)
        sal = predict(predictor, de, DEVICE) if any(m in M.USES_SALIENCE for m, _, _ in jobs) else {}
        print(f"\n=== {args.model} / {args.bench} / {split} : {len(de)} pages, natif {nat:.1f} ===")

        n_orig = sum(e.shape[0] for e in de.values())
        for m, tag, k_of in jobs:
            backend = M.default_backend(m, P.dim) if args.kmeans == "auto" else args.kmeans
            t0 = time.time()
            method = M.build(m, de, profiles, DEVICE, seed=args.seed, backend=backend)
            ks = {did: k_of(e.shape[0]) for did, e in de.items()}
            if method.allocate:
                ks = method.allocate(ks)
            comp = {did: method.compress(de[did].to(DEVICE).float(), sal.get(did), ks[did]).cpu()
                    for did in tqdm(de, desc=f"{m} {tag}", leave=False)}
            res = evaluate(qe, comp, rel, DEVICE)
            n_kept = sum(c.shape[0] for c in comp.values())
            save_pickle(P.result(args.bench, split, tag, m), {
                "split": split, "method": m, "budget": tag, **res,
                "avg_vectors": n_kept / len(de), "compression": n_kept / n_orig,
                "time": time.time() - t0, "seed": args.seed, "backend": backend})
            print(f"  {m:<12} {tag:<6} nDCG@5={res['ndcg5']:5.1f}  rétention={100 * res['ndcg5'] / nat:5.1f} %"
                  f"  vecteurs/page={n_kept / len(de):6.1f}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
