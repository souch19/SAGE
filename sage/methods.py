"""Méthodes évaluées par 04_evaluate/evaluate.py, instanciées pour un split donné."""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from sage import operators as op

# "paper" : opérateurs du papier (S-Top-K, S-Group, Sage, S-Xr, S-Xp) ; les autres groupes sont des
# baselines exploratoires, hors papier, lancées seulement si on les demande.
GROUPS = {
    "paper":     ["select", "stride", "wkmeans", "XR_wkmeans", "XP_wkmeans"],
    "main":      ["select", "stride", "wkmeans", "XR_wkmeans"],
    "ablation":  ["kmeans", "ptm_mean", "ptm_wavg"],
    "trainfree": ["random", "bottomk_red", "norm_sel", "profile_sel", "profile_wkm", "dyn_wkmeans"],
    "adaptive":  ["omni_select", "omni_merge"],
}
ALL = list(dict.fromkeys(m for g in GROUPS.values() for m in g))
USES_SALIENCE = {"select", "stride", "wkmeans", "XR_wkmeans", "XP_wkmeans", "ptm_mean", "ptm_wavg"}
USES_PROFILES = {"stride", "XR_wkmeans", "norm_sel", "profile_sel", "profile_wkm", "dyn_wkmeans"}


def expand(names):
    """Noms de méthodes ou de groupes ('main', 'all'...) -> liste de méthodes sans doublon."""
    out = []
    for n in names:
        for m in ALL if n == "all" else GROUPS.get(n, [n]):
            if m not in ALL:
                raise SystemExit(f"méthode inconnue '{m}' (méthodes : {', '.join(ALL)} ; "
                                 f"groupes : {', '.join(GROUPS)}, all)")
            if m not in out:
                out.append(m)
    return out


def default_backend(method, dim):
    """Backend k-means utilisé pour les résultats du papier."""
    if method in ("XR_wkmeans", "XP_wkmeans") or (method == "wkmeans" and dim > 128):
        return "gpu"
    return "sklearn"


@dataclass
class Method:
    compress: Callable                    # (p, sal, k) -> [k', D]
    allocate: Optional[Callable] = None   # ({page: k}) -> {page: k} (budget adaptatif)


def build(name, de, profiles, device, seed=42, backend="sklearn", beta=1.0, theta=0.85):
    """Instancie `name` pour les pages `de` d'un split (les calibrages éventuels se font ici)."""
    if name == "select":
        return Method(op.select_topk)
    if name == "stride":
        eps = op.calibrate_eps(de, profiles, device, seed=seed)
        return Method(lambda p, s, k: op.s_group(p, s, k, profiles, eps))
    if name == "wkmeans":
        return Method(lambda p, s, k: op.wkmeans(p, s, k, backend, seed))
    if name == "kmeans":
        return Method(lambda p, s, k: op.wkmeans(p, None, k, backend, seed))
    if name == "XR_wkmeans":
        return Method(lambda p, s, k: op.xr_wkmeans(p, s, k, profiles, beta, backend, seed))
    if name == "XP_wkmeans":
        return Method(lambda p, s, k: op.xp_wkmeans(p, s, k, beta, backend, seed))
    if name in ("ptm_mean", "ptm_wavg"):
        return Method(lambda p, s, k: op.prune_then_merge(p, s, k, weighted=name == "ptm_wavg"))
    if name == "random":
        gen = torch.Generator().manual_seed(seed)
        return Method(lambda p, s, k: op.random_subset(p, k, gen))
    if name == "bottomk_red":
        return Method(lambda p, s, k: op.least_redundant(p, k))
    if name == "norm_sel":
        return Method(lambda p, s, k: op.select_topk(p, op.profile_energy(p, profiles), k))
    if name == "profile_sel":
        return Method(lambda p, s, k: op.select_topk(p, op.profile_max(p, profiles), k))
    if name == "profile_wkm":
        return Method(lambda p, s, k: op.wkmeans(p, op.profile_max(p, profiles), k, backend, seed))
    if name == "dyn_wkmeans":
        # page concentrée (CV faible) -> sélection, page diffuse -> fusion ; seuil = CV médian du split
        cvs = [op.coeff_var(op.profile_max(de[d].to(device).float(), profiles)) for d in de]
        tau = float(np.quantile(cvs, 0.5))

        def route(p, s, k):
            r = op.profile_max(p, profiles)
            return op.select_topk(p, r, k) if op.coeff_var(r) <= tau else op.wkmeans(p, r, k, backend, seed)
        return Method(route)
    if name in ("omni_select", "omni_merge"):
        fn = op.omni_select if name == "omni_select" else op.omni_merge

        def allocate(ks):
            dids = list(ks)
            n = [de[d].shape[0] for d in dids]
            groups = [op.threshold_groups(de[d].to(device).float(), theta)[1] for d in dids]
            kd = op.normalize_budgets(groups, n, sum(ks.values()))
            return {d: int(min(k, nd)) for d, k, nd in zip(dids, kd, n)}
        return Method(lambda p, s, k: fn(p, k, theta, seed), allocate)
    raise ValueError(name)
