"""Scoring MaxSim (late interaction) et métriques de rang : nDCG@5, succès@k, valeurs par requête."""
import numpy as np
import torch

from sage.utils import load_pickle, save_pickle


@torch.inference_mode()
def maxsim(qe, de, qids, device, chunk=128):
    """{qid: {page: score}} ; les pages sont regroupées par paquets de `chunk` et paddées."""
    dids, scores = list(de), {q: {} for q in qids}
    for c0 in range(0, len(dids), chunk):
        cids = dids[c0:c0 + chunk]
        lens = [de[d].shape[0] for d in cids]
        P = torch.zeros(len(cids), max(lens), de[cids[0]].shape[1], device=device)
        mask = torch.zeros(len(cids), max(lens), dtype=torch.bool, device=device)
        for i, d in enumerate(cids):
            P[i, :lens[i]] = de[d].to(device).float()
            mask[i, :lens[i]] = True
        neg = torch.finfo(P.dtype).min
        for q in qids:
            sim = torch.einsum("td,nld->tnl", qe[q].to(device).float(), P)
            s = sim.masked_fill(~mask.unsqueeze(0), neg).max(-1).values.sum(0)
            scores[q].update(zip(cids, s.tolist()))
    return scores


def rank_metrics(scores, rel, ks=(1, 5, 10)):
    """nDCG@5 x100 (moyenne sur les requêtes), succès@k, et nDCG@5 de chaque requête."""
    hits, per_query, total = dict.fromkeys(ks, 0), {}, 0.0
    for q, pos in rel.items():
        ranked = [d for d, _ in sorted(scores[q].items(), key=lambda x: x[1], reverse=True)[:max(ks)]]
        for k in ks:
            hits[k] += bool(set(ranked[:k]) & pos)
        dcg = sum(1 / np.log2(r + 2) for r, d in enumerate(ranked[:5]) if d in pos)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(pos), 5)))
        v = dcg / idcg if idcg > 0 else 0.0
        total += v
        per_query[q] = float(100 * v)
    n = len(rel)
    return {"ndcg5": float(total / n * 100),
            "recall": {k: hits[k] / n for k in ks},
            "per_query": per_query}


def evaluate(qe, de, rel, device):
    if not rel:
        raise ValueError("aucune requête n'a de page pertinente dans ce corpus")
    return rank_metrics(maxsim(qe, de, list(rel), device), rel)


def native_ndcg(paths, bench, split, emb, device):
    """nDCG@5 de l'index complet d'un split, calculé une fois puis lu dans native.pkl."""
    f = paths.native(bench, split)
    if not f.exists():
        res = evaluate(emb["q_emb"], emb["doc_emb"], emb["rel"], device)
        save_pickle(f, {"split": split, "method": "native", **res})
    return load_pickle(f)["ndcg5"]
