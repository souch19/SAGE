"""Encode les splits d'un banc d'évaluation et calcule le nDCG@5 natif (100 % des vecteurs).

Sortie : $SAGE_ARTIFACTS/{modèle}/embeddings/{banc}/{split}.pkl
         {doc_emb, q_emb, rel, ndcg5, npatch} ; un split déjà encodé est sauté.

    python 02_encode/encode_eval.py --model colqwen --bench vidore_v2
    python 02_encode/encode_eval.py --model colqwen --bench jinavdr --splits slide mmtab
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.data import Split
from sage.encoders import build_encoder
from sage.retrieval import evaluate
from sage.utils import DEVICE, load_pickle, save_pickle


def encode_split(enc, split, limit=None):
    doc_ids = split.doc_ids
    if limit:   # pages pertinentes d'abord, pour garder des requêtes évaluables
        relevant = {d for pos in split.rel.values() for d in pos}
        doc_ids = sorted(doc_ids, key=lambda d: d not in relevant)[:limit]
    de = {d: enc.encode_doc(split.image(d)) for d in tqdm(doc_ids, desc="pages")}
    rel = {q: {d for d in pos if d in de} for q, pos in split.rel.items() if q in split.queries}
    rel = {q: pos for q, pos in rel.items() if pos}
    qe = {q: enc.encode_query(split.queries[q]) for q in tqdm(rel, desc="requêtes")}
    return de, qe, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--bench", default="vidore_v2", choices=list(C.BENCHMARKS))
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="n'encoder que les N premières pages (test)")
    args = ap.parse_args()

    P = C.Paths(args.model)
    todo = {s: r for s, r in C.splits(args.bench, args.splits).items()
            if not P.emb(args.bench, s).exists()}
    enc = build_encoder(args.model) if todo else None
    for name, repo in todo.items():
        print(f"\n=== {args.model} / {args.bench} / {name} ({repo}) ===")
        de, qe, rel = encode_split(enc, Split(args.bench, repo), args.limit)
        if not rel:
            print("  [!] aucune requête avec une page pertinente encodée, split ignoré")
            continue
        nd = evaluate(qe, de, rel, DEVICE)["ndcg5"]
        npatch = int(np.mean([e.shape[0] for e in de.values()]))
        save_pickle(P.emb(args.bench, name),
                    {"doc_emb": de, "q_emb": qe, "rel": rel, "ndcg5": nd, "npatch": npatch})
        print(f"  natif nDCG@5={nd:.1f} | pages={len(de)} requêtes={len(rel)} vecteurs/page={npatch}")
        del de, qe
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nnDCG@5 natif, {args.model} / {args.bench}")
    for name in C.splits(args.bench, args.splits):
        if P.emb(args.bench, name).exists():
            d = load_pickle(P.emb(args.bench, name))
            print(f"  {name:<12} {d['ndcg5']:5.1f}  ({len(d['doc_emb'])} pages, {d['npatch']} vecteurs/page)")


if __name__ == "__main__":
    main()
