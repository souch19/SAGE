"""Opérateurs de compression d'une page : (patchs p [N, D], saillance [N], budget k) -> [k', D].

Nom de code      Papier
select           S-Top-K   top-k des patchs les plus saillants
stride           S-Group   regroupement glouton (voisinage spatial et de réponse) + fusion softmax
wkmeans          SAGE      k-means pondéré par la saillance
XR_wkmeans       S-Xr      k-means pondéré sur [beta X (+) R], X position, R réponse aux profils
XP_wkmeans       S-Xp      k-means pondéré sur [beta X (+) P], P embedding du patch
kmeans                     k-means uniforme (ablation de la saillance)
ptm_mean/_wavg             Prune-then-Merge (Yan et al.) : seuil adaptatif puis Ward
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from sage.kmeans import cluster


def positions_2d(n, device):
    gw = int(np.ceil(np.sqrt(n)))
    idx = torch.arange(n, device=device)
    return torch.stack([(idx // gw).float(), (idx % gw).float()], -1)


def responses(p, profiles):
    """Profil de réponse R de chaque patch aux tokens des requêtes profils, L2-normalisé."""
    return F.normalize(p @ profiles.to(p.device).t(), dim=-1)


def _zscore(x):
    z = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return z / math.sqrt(z.shape[1])


def _weighted_merge(p, labels, w, k):
    """Centroïdes des groupes, moyennes de p pondérées par w ; groupes vides retirés."""
    D = p.shape[1]
    den = torch.zeros(k, device=p.device).scatter_add_(0, labels, w)
    Z = torch.zeros(k, D, device=p.device, dtype=p.dtype)
    Z.scatter_add_(0, labels.unsqueeze(-1).expand(-1, D), p * w.unsqueeze(-1))
    Z = Z / den.clamp(min=1e-8).unsqueeze(-1)
    return F.normalize(Z[den > 0], dim=-1)


def select_topk(p, sal, k):
    return F.normalize(p[sal.topk(min(k, p.shape[0])).indices], dim=-1)


def s_group(p, sal, k, profiles, eps, radius=2.0, tau=0.1):
    """Ancres prises par saillance décroissante ; chaque ancre absorbe les patchs libres
    proches en position (< radius) et en réponse (< eps), fusionnés par softmax(sal/tau)."""
    N, dev = p.shape[0], p.device
    pos = positions_2d(N, dev)
    prof = responses(p, profiles)
    adj = (torch.cdist(pos, pos) < radius) & (torch.cdist(prof, prof) < eps)
    gid = torch.full((N,), -1, dtype=torch.long, device=dev)
    used = torch.zeros(N, dtype=torch.bool, device=dev)
    ng = 0
    for a in sal.argsort(descending=True).tolist():
        if used[a]:
            continue
        nb = adj[a] & ~used
        gid[nb] = ng
        used |= nb
        ng += 1
    logits = sal / tau
    m = torch.full((ng,), float("-inf"), device=dev).scatter_reduce_(
        0, gid, logits, reduce="amax", include_self=False)
    ex = torch.exp(logits - m[gid])
    alpha = ex / torch.zeros(ng, device=dev).scatter_add_(0, gid, ex).clamp(min=1e-8)[gid]
    Z = torch.zeros(ng, p.shape[1], device=dev, dtype=p.dtype).scatter_add_(
        0, gid.unsqueeze(-1).expand(-1, p.shape[1]), p * alpha.unsqueeze(-1))
    mass = torch.zeros(ng, device=dev).scatter_add_(0, gid, sal)
    if ng > min(k, N):
        Z = Z[mass.topk(min(k, N)).indices]
    return F.normalize(Z, dim=-1)


def calibrate_eps(de, profiles, device, quantile=0.30, n_docs=16, seed=42):
    """Seuil de réponse de S-Group : quantile des distances entre profils de paires de patchs
    tirées au hasard dans quelques pages du split (générateurs locaux : indépendant de l'ordre des runs)."""
    import random
    docs = random.Random(seed).sample(list(de), min(n_docs, len(de)))
    g = torch.Generator().manual_seed(seed)
    dists = []
    with torch.no_grad():
        for d in docs:
            r = responses(de[d].to(device).float(), profiles)
            i = torch.randint(0, r.shape[0], (512,), generator=g)
            j = torch.randint(0, r.shape[0], (512,), generator=g)
            dists.append((r[i] - r[j]).norm(dim=-1).cpu())
    return torch.cat(dists).quantile(quantile).item()


def wkmeans(p, sal, k, backend="sklearn", seed=42):
    """Centroïdes du k-means pondéré par la saillance (uniforme si sal est None)."""
    w = None if sal is None else sal.clamp(min=1e-4)
    C, _ = cluster(p, min(k, p.shape[0]), w, backend, seed)
    return F.normalize(C, dim=-1)


def feature_wkmeans(p, sal, k, feats, backend="sklearn", seed=42):
    """Partition par k-means pondéré dans l'espace `feats`, puis agrégation de p pondérée par sal."""
    k, w = min(k, p.shape[0]), sal.clamp(min=1e-4)
    _, labels = cluster(feats, k, w, backend, seed)
    return _weighted_merge(p, labels, w, k)


def xr_wkmeans(p, sal, k, profiles, beta=1.0, backend="gpu", seed=42):
    X = _zscore(positions_2d(p.shape[0], p.device))
    R = _zscore(responses(p, profiles))
    return feature_wkmeans(p, sal, k, torch.cat([beta * X, R], -1), backend, seed)


def xp_wkmeans(p, sal, k, beta=1.0, backend="gpu", seed=42):
    X = _zscore(positions_2d(p.shape[0], p.device))
    return feature_wkmeans(p, sal, k, torch.cat([beta * X, _zscore(p)], -1), backend, seed)


def prune_then_merge(p, sal, k, weighted=False, prune_k=0.0):
    """Garde les patchs de saillance >= mu + prune_k sigma (au moins k), puis Ward jusqu'à k groupes."""
    from scipy.cluster import hierarchy
    keep = sal >= sal.mean() + prune_k * sal.std()
    if keep.sum() < k:
        keep = torch.zeros_like(keep)
        keep[sal.topk(min(k, p.shape[0])).indices] = True
    p, sal = p[keep], sal[keep]
    N, k = p.shape[0], min(k, p.shape[0])
    if N <= k:
        return F.normalize(p, dim=-1)
    dist = 1.0 - (p @ p.t()).float().cpu().numpy()
    np.fill_diagonal(dist, 0.0)
    Z = hierarchy.linkage(dist[np.triu_indices(N, k=1)].clip(min=0.0), method="ward")
    labels = torch.from_numpy(hierarchy.fcluster(Z, t=k, criterion="maxclust") - 1).long().to(p.device)
    w = sal.clamp(min=1e-4) if weighted else torch.ones_like(sal)
    return _weighted_merge(p, labels, w, int(labels.max()) + 1)


# --- signaux sans apprentissage (baselines type OmniSelect) ---

def profile_max(p, profiles):
    """Réponse maximale du patch aux tokens profils (saillance zero-shot)."""
    return (p @ profiles.to(p.device).t()).max(dim=-1).values


def profile_energy(p, profiles):
    return (p @ profiles.to(p.device).t()).pow(2).sum(dim=-1)


def coeff_var(x):
    return (x.std() / x.mean().clamp(min=1e-6)).item()


def random_subset(p, k, generator):
    idx = torch.randperm(p.shape[0], generator=generator, device="cpu")[:min(k, p.shape[0])]
    return F.normalize(p[idx.to(p.device)], dim=-1)


def least_redundant(p, k):
    """Les k patchs de plus faible similarité moyenne aux autres patchs de la page."""
    N = p.shape[0]
    red = ((p @ p.t()).sum(dim=-1) - 1.0) / max(N - 1, 1)
    return F.normalize(p[red.topk(min(k, N), largest=False).indices], dim=-1)


# --- budget adaptatif par page (allocation type OmniSelect, sans requête) ---

def threshold_groups(p, theta):
    """Composantes connexes du graphe cos(p_i, p_j) > theta : (labels [N], nb de groupes)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    ng, lab = connected_components(csr_matrix(((p @ p.t()) > theta).cpu().numpy()), directed=False)
    return torch.from_numpy(lab).long().to(p.device), int(ng)


def normalize_budgets(kd, n, total, kd_min=8, iters=50):
    """Remet à l'échelle les budgets bruts pour que leur somme vaille `total`, avec kd_min <= kd <= n."""
    n = np.asarray(n, dtype=float)
    kd = np.clip(np.asarray(kd, dtype=float), kd_min, n)
    for _ in range(iters):
        if abs(kd.sum() - total) < 1.0:
            break
        free = (kd > kd_min) & (kd < n)
        if not free.any():
            break
        scale = (total - kd[~free].sum()) / max(kd[free].sum(), 1e-8)
        kd[free] = np.clip(kd[free] * scale, kd_min, n[free])
    return np.maximum(1, np.round(kd)).astype(int)


def _groups_to_k(p, k, theta, seed):
    labels, ng = threshold_groups(p, theta)
    if ng <= k:
        return labels, ng
    D = p.shape[1]
    C = torch.zeros(ng, D, device=p.device).scatter_add_(0, labels.unsqueeze(-1).expand(-1, D), p)
    cnt = torch.zeros(ng, device=p.device).scatter_add_(0, labels, torch.ones_like(labels, dtype=p.dtype))
    C = F.normalize(C / cnt.clamp(min=1).unsqueeze(-1), dim=-1)
    _, lab2 = cluster(C, k, None, "sklearn", seed)
    return lab2[labels], k


def omni_select(p, k, theta=0.85, seed=42):
    """Un représentant par groupe : le patch le plus proche du centroïde du groupe."""
    labels, kk = _groups_to_k(p, min(k, p.shape[0]), theta, seed)
    reps = []
    for g in range(kk):
        members = (labels == g).nonzero(as_tuple=True)[0]
        if members.numel():
            centroid = F.normalize(p[members].mean(0, keepdim=True), dim=-1)
            reps.append(members[(p[members] @ centroid.t()).squeeze(-1).argmax()].item())
    return F.normalize(p[torch.tensor(reps, device=p.device)], dim=-1)


def omni_merge(p, k, theta=0.85, seed=42):
    """Un centroïde (moyenne simple) par groupe."""
    labels, kk = _groups_to_k(p, min(k, p.shape[0]), theta, seed)
    return _weighted_merge(p, labels, torch.ones(p.shape[0], device=p.device), kk)
