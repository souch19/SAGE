"""k-means pondéré, deux backends à sémantique identique (sample_weight de sklearn) :
    sklearn : KMeans(n_init=3), CPU
    gpu     : k-means++ pondéré + Lloyd en torch (tourne aussi sur CPU)
"""
import torch
from sklearn.cluster import KMeans

N_INIT, MAX_ITER, TOL = 3, 300, 1e-4


@torch.no_grad()
def kmeans_pp_init(X, k, w, generator):
    n = X.shape[0]
    C = X[torch.multinomial(w, 1, generator=generator)]
    d2 = ((X - C[0]) ** 2).sum(-1)
    for _ in range(k - 1):
        p = (d2 * w).clamp(min=0)
        s = p.sum()
        idx = (torch.multinomial(p / s, 1, generator=generator) if s > 0
               else torch.randint(0, n, (1,), device=X.device, generator=generator))
        C = torch.cat([C, X[idx]], 0)
        d2 = torch.minimum(d2, ((X - X[idx][0]) ** 2).sum(-1))
    return C


@torch.no_grad()
def torch_kmeans(X, k, w=None, n_init=N_INIT, max_iter=MAX_ITER, tol=TOL, seed=42):
    """Renvoie (centroïdes, nb d'itérations, inertie) du meilleur des n_init essais."""
    n, d = X.shape
    k = min(k, n)
    w = torch.ones(n, device=X.device) if w is None else w.clamp(min=1e-8)
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    best_C, best_in, best_it = None, float("inf"), 0
    for _ in range(n_init):
        C, prev = kmeans_pp_init(X, k, w, g), None
        for it in range(max_iter):
            d2 = torch.cdist(X, C).pow_(2)
            lab = d2.argmin(1)
            Cn = torch.zeros_like(C).scatter_add_(0, lab.unsqueeze(-1).expand(-1, d), X * w.unsqueeze(-1))
            den = torch.zeros(k, device=X.device).scatter_add_(0, lab, w)
            empty = den <= 1e-12
            Cn = Cn / den.clamp(min=1e-12).unsqueeze(-1)
            if empty.any():      # cluster vide : réinitialisé sur les points les plus éloignés
                far = d2.gather(1, lab.unsqueeze(1)).squeeze(1).topk(int(empty.sum())).indices
                Cn[empty] = X[far]
            shift = (Cn - C).pow(2).sum()
            C = Cn
            if prev is not None and shift <= tol:
                break
            prev = shift
        inertia = (torch.cdist(X, C).pow_(2).min(1).values * w).sum().item()
        if inertia < best_in:
            best_C, best_in, best_it = C.clone(), inertia, it + 1
    return best_C, best_it, best_in


def cluster(X, k, w=None, backend="sklearn", seed=42):
    """Centroïdes [k, d] et affectation de chaque point [n] (int64), sur le device de X."""
    if backend == "gpu":
        C, _, _ = torch_kmeans(X, k, w, seed=seed)
        return C, torch.cdist(X, C).argmin(1)
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(
        X.cpu().numpy(), sample_weight=None if w is None else w.cpu().numpy())
    return (torch.from_numpy(km.cluster_centers_).float().to(X.device),
            torch.from_numpy(km.labels_).long().to(X.device))
