"""Borne haute query-aware (Top-K conditionné par la requête, à la OmniSelect).

Pour chaque couple (requête, page), on garde les k patchs de plus forte similarité maximale
aux tokens de la requête, puis on calcule le MaxSim sur ces k patchs. La sélection dépend de
la requête : ce n'est pas un index compressé, seulement un point de comparaison.
Sortie : results/{banc}/{split}/{budget}/topk_query.pkl

    python 04_evaluate/query_conditioned.py --model colbert --budgets 05pct 64abs
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.retrieval import native_ndcg, rank_metrics
from sage.utils import DEVICE, load_pickle, save_pickle


@torch.inference_mode()
def pruned_scores(qe, de, qids, k_of, chunk=64):
    scores = {q: {} for q in qids}
    dids = list(de)
    for c0 in tqdm(range(0, len(dids), chunk), leave=False):
        P = {d: de[d].to(DEVICE).float() for d in dids[c0:c0 + chunk]}
        for q in qids:
            Q = qe[q].to(DEVICE).float()
            for d, p in P.items():
                sim = Q @ p.t()
                keep = sim.max(dim=0).values.topk(k_of(p.shape[0])).indices
                scores[q][d] = sim[:, keep].max(dim=-1).values.sum().item()
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--bench", default="vidore_v2", choices=list(C.BENCHMARKS))
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--budgets", nargs="*", default=["05pct", "64abs"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    P = C.Paths(args.model)
    budgets = [C.parse_budget(b) for b in args.budgets]
    for split in C.splits(args.bench, args.splits):
        todo = [(t, k_of) for t, k_of in budgets
                if args.force or not P.result(args.bench, split, t, "topk_query").exists()]
        if not todo or not P.emb(args.bench, split).exists():
            continue
        d = load_pickle(P.emb(args.bench, split))
        nat = native_ndcg(P, args.bench, split, d, DEVICE)
        for tag, k_of in todo:
            t0 = time.time()
            res = rank_metrics(pruned_scores(d["q_emb"], d["doc_emb"], list(d["rel"]), k_of), d["rel"])
            avg = sum(k_of(e.shape[0]) for e in d["doc_emb"].values()) / len(d["doc_emb"])
            save_pickle(P.result(args.bench, split, tag, "topk_query"), {
                "split": split, "method": "topk_query", "budget": tag, **res,
                "avg_vectors": avg, "time": time.time() - t0})
            print(f"[{args.model}/{split}] topk_query {tag:<6} nDCG@5={res['ndcg5']:5.1f}"
                  f"  rétention={100 * res['ndcg5'] / nat:5.1f} %")


if __name__ == "__main__":
    main()
