"""Encode le pool d'entraînement (pages et requêtes) avec le modèle choisi.

Sorties dans $SAGE_ARTIFACTS/{modèle}/train/ : docs/shard_{i}.pkl (2000 pages par shard, fp16),
queries.pkl, meta.pkl. Un shard déjà présent est sauté.

    python 02_encode/encode_train.py --model colqwen
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.data import train_pool
from sage.encoders import build_encoder
from sage.utils import save_pickle

SHARD = 2000


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--data", default=str(C.DATA), help="dossier du pool (01_data/build_train10k.py)")
    ap.add_argument("--limit", type=int, default=None, help="n'encoder que les N premières pages (test)")
    args = ap.parse_args()

    P = C.Paths(args.model)
    pages, qid2query, rel, resolve = train_pool(args.data)
    if args.limit:
        pages = pages[:args.limit]
        kept = set(pages)
        rel = {q: v for q, v in rel.items() if all(d in kept for d in v)}
        qid2query = {q: t for q, t in qid2query.items() if q in rel}
    n_shards = (len(pages) + SHARD - 1) // SHARD
    print(f"[{args.model}] pages={len(pages)} requêtes={len(qid2query)} shards={n_shards}")

    todo = [i for i in range(n_shards) if not P.train_docs(i).exists()]
    need_queries = not P.train("queries.pkl").exists()
    if todo or need_queries:
        enc = build_encoder(args.model)
        for i in todo:
            de, sizes = {}, []
            for imf in tqdm(pages[i * SHARD:(i + 1) * SHARD], desc=f"shard {i + 1}/{n_shards}"):
                path = resolve(imf)
                if path is None:
                    continue
                try:
                    de[imf] = enc.encode_doc(Image.open(path))
                    sizes.append(de[imf].shape[0])
                except Exception as e:
                    print(f"  [!] {imf[:60]} : {e}")
            save_pickle(P.train_docs(i), {"docs": de})
            print(f"  shard {i} : {len(de)} pages, {int(np.mean(sizes)) if sizes else 0} vecteurs/page")
        if need_queries:
            qe = {q: enc.encode_query(t) for q, t in tqdm(qid2query.items(), desc="requêtes")}
            save_pickle(P.train("queries.pkl"), {"queries": qe})

    save_pickle(P.train("meta.pkl"), {"rel": rel, "qid2query": qid2query,
                                      "images": pages, "n_shards": n_shards})
    print(f"[{args.model}] pool encodé -> {P.root / 'train'}")


if __name__ == "__main__":
    main()
