"""Cibles de saillance par attribution MaxSim sur le pool d'entraînement.

Pour chaque page et chacune de ses requêtes : hard = vote argmax de chaque token requête
(normalisé à somme 1), soft = masse softmax(sim / 0.05) (normalisée à max 1).
Sortie : train/targets.pkl {targets: {page: {hard, soft}}, common_doc_ids}.

    python 03_salience/attribution.py --model colqwen
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.utils import DEVICE, load_pickle, save_pickle

TAU = 0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    args = ap.parse_args()

    P = C.Paths(args.model)
    if P.train("targets.pkl").exists():
        print(f"[{args.model}] cibles déjà calculées")
        return
    meta = load_pickle(P.train("meta.pkl"))
    qtok = load_pickle(P.train("queries.pkl"))["queries"]
    doc2qids = defaultdict(list)
    for qid, docs in meta["rel"].items():
        if qid in qtok:
            for d in docs:
                doc2qids[d].append(qid)
    print(f"[{args.model}] requêtes={len(qtok)} pages référencées={len(doc2qids)}")

    hard, soft = {}, {}
    for i in range(meta["n_shards"]):
        shard = load_pickle(P.train_docs(i))["docs"]
        for did, emb in tqdm(shard.items(), desc=f"shard {i + 1}/{meta['n_shards']}"):
            if did not in doc2qids:
                continue
            p = emb.to(DEVICE).float()
            h, s = torch.zeros(p.shape[0]), torch.zeros(p.shape[0])
            for qid in doc2qids[did]:
                with torch.inference_mode():
                    sim = qtok[qid].to(DEVICE).float() @ p.t()
                    am = sim.argmax(dim=-1)
                    h.scatter_add_(0, am.cpu(), torch.full((len(am),), 1.0 / max(len(am), 1)))
                    s += torch.softmax(sim / TAU, dim=-1).sum(0).cpu()
            hard[did], soft[did] = h, s
        del shard
        torch.cuda.empty_cache()

    targets = {}
    for d in hard:
        ts = soft[d] / soft[d].max().clamp(min=1e-8)
        if ts.sum() > 0:
            targets[d] = {"hard": hard[d] / hard[d].sum().clamp(min=1e-8), "soft": ts}
    save_pickle(P.train("targets.pkl"), {"targets": targets, "common_doc_ids": list(targets)})
    dens = {k: 100 * np.mean([(t[k] > 0.01).float().mean().item() for t in targets.values()])
            for k in ("hard", "soft")}
    print(f"[{args.model}] cibles pour {len(targets)} pages | densité >0.01 : "
          f"hard {dens['hard']:.1f} %, soft {dens['soft']:.0f} %")


if __name__ == "__main__":
    main()
