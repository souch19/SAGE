"""Entraîne le régresseur de saillance (MLP + perte de rang) sur les cibles soft.

Les shards sont copiés sur le disque local (--scratch) puis lus un par un. Sorties :
train/regressor[_s{seed}].pt et .meta.json ; affiche la corrélation de Spearman intra-page
sur 15 % de pages tenues à l'écart.

    python 03_salience/train_regressor.py --model colqwen [--seed 1] [--cleanup]
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.salience import HIDDEN, SaliencyMLP
from sage.utils import DEVICE, load_pickle, save_json, save_torch, seed_everything

LR, EPOCHS, BATCH, RANK_W, KEY = 1e-3, 60, 8, 0.5, "soft"
VAL_FRAC, TEST_FRAC = 0.15, 0.15


class ShardDataset(Dataset):
    """Pages triées par shard : un seul shard en mémoire à la fois. Padding à max_len."""

    def __init__(self, ids, doc2shard, shard_path, targets, max_len):
        self.items = sorted(ids, key=lambda d: (doc2shard[d], d))
        self.doc2shard, self.shard_path, self.targets, self.max_len = doc2shard, shard_path, targets, max_len
        self._idx, self._docs = None, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        did = self.items[i]
        if self._idx != self.doc2shard[did]:
            self._idx = self.doc2shard[did]
            self._docs = load_pickle(self.shard_path(self._idx))["docs"]
        p, s = self._docs[did].float(), self.targets[did][KEY].float()
        n, pad = p.shape[0], self.max_len - p.shape[0]
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:n] = True
        if pad > 0:
            p, s = F.pad(p, (0, 0, 0, pad)), F.pad(s, (0, pad))
        elif pad < 0:
            p, s, mask = p[:self.max_len], s[:self.max_len], mask[:self.max_len]
        return p, s, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"), help="disque local rapide")
    ap.add_argument("--cleanup", action="store_true", help="supprime les shards d'entraînement après coup")
    args = ap.parse_args()

    P = C.Paths(args.model, args.seed)
    if P.regressor().exists() and P.regressor_meta().exists():
        print(f"[{args.model}] {P.regressor().name} existe déjà")
        return
    seed_everything(args.seed)
    meta = load_pickle(P.train("meta.pkl"))
    n_shards = meta["n_shards"]
    scratch = Path(args.scratch) / f"sage_{args.model}_shards"
    scratch.mkdir(parents=True, exist_ok=True)
    local = lambda i: scratch / f"shard_{i}.pkl"
    for i in range(n_shards):
        if not (local(i).exists() and local(i).stat().st_size == P.train_docs(i).stat().st_size):
            shutil.copy2(P.train_docs(i), local(i))

    targets = load_pickle(P.train("targets.pkl"))["targets"]
    doc2shard, max_len, dim = {}, 0, P.dim
    for i in range(n_shards):
        for did, p in load_pickle(local(i))["docs"].items():
            if did in targets:
                doc2shard[did], max_len = i, max(max_len, p.shape[0])
            dim = p.shape[1]
    usable = sorted(doc2shard)
    perm = np.random.default_rng(args.seed).permutation(len(usable))
    n_test, n_val = int(TEST_FRAC * len(usable)), int(VAL_FRAC * len(usable))
    test_ids = [usable[i] for i in perm[:n_test]]
    train_ids = [usable[i] for i in perm[n_test + n_val:]]
    print(f"[{args.model}] seed={args.seed} train={len(train_ids)} test={len(test_ids)} "
          f"dim={dim} max_len={max_len}")

    torch.manual_seed(args.seed)
    model = SaliencyMLP(dim, HIDDEN).to(DEVICE)
    loader = DataLoader(ShardDataset(train_ids, doc2shard, local, targets, max_len),
                        batch_size=BATCH, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    margin = nn.MarginRankingLoss(margin=0.1)
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for bp, bs, bm in loader:
            bp, bs, valid = bp.to(DEVICE), bs.to(DEVICE), bm.to(DEVICE).float()
            pred = model(bp)
            fit = ((pred - bs) ** 2 * valid).sum() / valid.sum().clamp(min=1)
            B, n = pred.shape
            i = torch.randint(0, n, (B, 256), device=DEVICE)
            j = torch.randint(0, n, (B, 256), device=DEVICE)
            vij = torch.gather(valid, 1, i) * torch.gather(valid, 1, j)
            sign = torch.sign(torch.gather(bs, 1, i) - torch.gather(bs, 1, j))
            # MarginRankingLoss en réduction 'mean' : le masque vij n'agit que globalement (réglage d'origine)
            rank = (margin(torch.gather(pred, 1, i).flatten(), torch.gather(pred, 1, j).flatten(),
                           sign.flatten()) * vij.flatten()).sum() / vij.sum().clamp(min=1)
            loss = fit + RANK_W * rank
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        sched.step()
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:>3} loss={tot / max(nb, 1):.4f}")
    train_time = time.time() - t0

    save_torch(P.regressor(), model.state_dict())
    save_json(P.regressor_meta(), {
        "model": args.model, "seed": args.seed, "embed_dim": dim, "hidden": HIDDEN, "key": KEY,
        "epochs": EPOCHS, "n_train_docs": len(train_ids), "train_time": train_time,
        "params": sum(p.numel() for p in model.parameters())})
    print(f"[{args.model}] {P.regressor().name} ({train_time / 60:.1f} min)")

    model.eval()
    rhos, cur, docs = [], None, None
    for did in sorted(test_ids, key=lambda d: doc2shard[d]):
        if doc2shard[did] != cur:
            cur = doc2shard[did]
            docs = load_pickle(local(cur))["docs"]
        n = docs[did].shape[0]
        with torch.no_grad():
            pred = model(docs[did].float().to(DEVICE).unsqueeze(0)).squeeze(0).cpu().numpy()[:n]
        rho = spearmanr(targets[did][KEY].float().numpy()[:n], pred).correlation
        if not np.isnan(rho):
            rhos.append(rho)
    print(f"Spearman intra-page (test) = {np.mean(rhos):.3f} ± {np.std(rhos):.3f}")

    if args.cleanup:
        for i in range(n_shards):
            P.train_docs(i).unlink(missing_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)
        print("shards d'entraînement supprimés")


if __name__ == "__main__":
    main()
